
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI


# 1. Define our fictional company documents
documents = [
    {
        "title": "Purchasing Policy",
        "content": (
            "Purchases over £25,000 require approval from both "
            "the department director and the Finance Director."
        )
    },
    {
        "title": "Annual Leave Policy",
        "content": (
            "Employees receive 25 days of annual leave per year, "
            "excluding public holidays."
        )
    },
    {
        "title": "Password Reset Procedure",
        "content": (
            "Employees must use the IT self-service portal "
            "to reset their passwords."
        )
    }
]


# 2. Connect to Databricks using our CLI profile
w = WorkspaceClient(profile="genai-lab")
client = DatabricksOpenAI(workspace_client=w)


# 3. Extract document content into a list of strings
texts = [document["content"] for document in documents]


# 4. Generate embeddings for all documents in one API call
response = client.embeddings.create(
    model="databricks-gte-large-en",
    input=texts
)


# 5. Attach each embedding to its original document
for document, result in zip(documents, response.data):
    document["embedding"] = result.embedding


# 6. Validate the document embeddings
#for document in documents:
 #   print(document["title"], len(document["embedding"]))


# 7. Generate an embedding for the user's question
question = "Who needs to approve a £30,000 expense?"

question_response = client.embeddings.create(
    model="databricks-gte-large-en",
    input=question
)

question_embedding = question_response.data[0].embedding

#print("Question embedding dimensions:", len(question_embedding))

import math

# Calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    return dot_product / (magnitude1 * magnitude2)


# Compare the question embedding with each document embedding
for document in documents:
    score = cosine_similarity(
        question_embedding,
        document["embedding"]
    )

 #   print(document["title"], round(score, 3))


# Store each document's similarity score
for document in documents:
    document["score"] = cosine_similarity(
        question_embedding,
        document["embedding"]
    )

# Sort documents by score, highest first
ranked_documents = sorted(
    documents,
    key=lambda document: document["score"],
    reverse=True
)

# Display the ranking
for document in ranked_documents:
    print(document["title"], round(document["score"], 3))


# Retrieve the highest-ranked document
best_document = ranked_documents[0]

# Reject documents that fall below our similarity threshold
threshold = 0.65

if best_document["score"] < threshold:
    print("No sufficiently relevant document found.")
    exit()
else:
    print("Relevant document found:", best_document["title"])

print("\nRetrieved document:", best_document["title"])
print("Retrieved context:", best_document["content"])


# 8. Generate an answer using the retrieved document
response = client.chat.completions.create(
    model="databricks-gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Northstar Analytics' knowledge assistant. "
                "Answer using only the supplied context. "
                "If the answer is not in the context, say 'I don't know'."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{best_document['content']}\n\n"
                f"Question: {question}"
            )
        }
    ],
    max_tokens=150
)

print("\nGenerated answer:", response.choices[0].message.content)