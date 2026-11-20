
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

# Configure MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/genai-rag-lab")

# Connect to Databricks
w = WorkspaceClient(profile="genai-lab")
client = DatabricksOpenAI(workspace_client=w)

MODEL = "databricks-gpt-oss-20b"
INDEX = "gen-ai-dev.rag_lab.documents_index"


# NEW: Capture retrieval as a child span
@mlflow.trace(name="retrieve_documents")
def retrieve_documents(question):
    return w.vector_search_indexes.query_index(
        index_name=INDEX,
        columns=["src_doc_id", "content"],
        query_text=question,
        query_type="HYBRID",
        num_results=3,
    )


# NEW: Capture the relevance decision as a child span
@mlflow.trace(name="check_relevance")
def check_relevance(question, context):
    system_prompt = (
        "Determine whether the supplied context contains enough "
        "information to answer the question. "
        "Reply with exactly YES or NO. "
        "Do not answer the question."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    # NEW: Record the exact prompts sent to the model
    mlflow.update_current_trace  # Not needed for span recording

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    decision = (
        response.choices[0].message.content or ""
    ).strip().upper()

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "decision": decision,
    }


# NEW: Capture answer generation as a child span
@mlflow.trace(name="generate_answer")
def generate_answer(question, context):
    system_prompt = (
        "Answer using only the supplied context. "
        "If the answer is not in the context, say 'I don't know'."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer = response.choices[0].message.content or ""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "answer": answer,
    }


# CHANGED: The real RAG workflow now lives inside
# one traced parent function
@mlflow.trace(name="rag_pipeline")
def run_rag(question):

    # Step 1: Retrieve documents
    search_response = retrieve_documents(question)

    print("\nAll retrieved documents:")

    rows = search_response.result.data_array or []

    for row in rows:
        print(f"{row[0]}: {row[2]:.3f}")

    if not rows:
        print("\nAnswer: I don't know.")
        return {
            "question": question,
            "answer": "I don't know.",
            "reason": "No documents retrieved",
        }

    # Step 2: Select the highest-ranked document
    top_result = rows[0]

    document_id = top_result[0]
    context = top_result[1]
    score = top_result[2]

    print("\nDocument:", document_id)
    print("Context:", context)
    print("Score:", score)

    # Step 3: Check relevance
    relevance_result = check_relevance(question, context)

    decision = relevance_result["decision"]

    print("\nRelevance check:", decision)

    if decision != "YES":
        print("\nAnswer: I don't know.")
        return {
            "question": question,
            "document_id": document_id,
            "relevance": decision,
            "answer": "I don't know.",
            "reason": "Insufficient context",
        }

    # Step 4: Generate the answer
    generation_result = generate_answer(question, context)

    answer = generation_result["answer"]

    print("\nAnswer:", answer)

    # NEW: Return a structured pipeline result
    return {
        "question": question,
        "document_id": document_id,
        "retrieval_score": score,
        "relevance": decision,
        "answer": answer,
    }


# CHANGED: One call executes the complete traced pipeline
question = "Who needs to approve a £30,000 expense?"
result = run_rag(question)
