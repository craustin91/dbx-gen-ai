from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

# 1. Connect to Databricks
w = WorkspaceClient(profile="genai-lab")
client = DatabricksOpenAI(workspace_client=w)

# 2. Define our fictional company policy
context = (
    "Northstar Analytics purchasing policy: "
    "Purchases over £25,000 require approval from both "
    "the department director and the Finance Director."
)

# 3. Define the user's question
question = "Does a £30,000 purchase require Finance Director approval?"

# 4. Call the foundation model
response = client.chat.completions.create(
    model="databricks-gpt-oss-20b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are Northstar Analytics' knowledge assistant. "
                "Answer questions using only the supplied context. "
                "If the context does not contain the answer, say 'I don't know'. "
                "Do not invent facts."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ],
    max_tokens=150
)

# 5. Print just the answer
print(response.choices[0].message.content)