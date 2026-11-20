
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

# Connect to Databricks
w = WorkspaceClient(profile="genai-lab")
client = DatabricksOpenAI(workspace_client=w)

question = "Who needs to approve a £30,000 expense?"

# Step 1: Retrieve relevant documents
search_response = w.vector_search_indexes.query_index(
    index_name="gen-ai-dev.rag_lab.documents_index",
    columns=["src_doc_id", "content"],
    query_text=question,
    query_type="HYBRID",
    num_results=3,
)

# Display retrieval results
print("\nAll retrieved documents:")

for row in search_response.result.data_array:
    print(f"{row[0]}: {row[2]:.3f}")

# Step 2: Extract the highest-ranked document
if not search_response.result.data_array:
    print("\nAnswer: I don't know.")
    raise SystemExit(0)

top_result = search_response.result.data_array[0]

document_id = top_result[0]
context = top_result[1]
score = top_result[2]

print("\nDocument:", document_id)
print("Context:", context)
print("Score:", score)

# Step 3: Check whether the context can answer the question
relevance_response = client.chat.completions.create(
    model="databricks-gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": (
                "Determine whether the supplied context contains enough "
                "information to answer the question. "
                "Reply with exactly YES or NO. "
                "Do not answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ],
)

relevance = (
    relevance_response.choices[0].message.content or ""
).strip().upper()

print("\nRelevance check:", relevance)

# Step 4: Stop if the context is insufficient
if relevance != "YES":
    print("\nAnswer: I don't know.")
    raise SystemExit(0)

# Step 5: Generate an answer using the retrieved context
response = client.chat.completions.create(
    model="databricks-gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer using only the supplied context. "
                "If the answer is not in the context, say 'I don't know'."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ],
)

print("\nAnswer:", response.choices[0].message.content)
