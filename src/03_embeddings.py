import math

from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

w = WorkspaceClient(profile="genai-lab")
client = DatabricksOpenAI(workspace_client=w)

text1 = "Who needs to approve a £30,000 expense?"
text2 = "How many days of annual leave do employees receive?"

response = client.embeddings.create(
    model="databricks-gte-large-en",
    input=[text1, text2]
)

vector1 = response.data[0].embedding
vector2 = response.data[1].embedding

# Calculate cosine similarity
dot_product = sum(a * b for a, b in zip(vector1, vector2))

magnitude1 = math.sqrt(sum(a * a for a in vector1))
magnitude2 = math.sqrt(sum(b * b for b in vector2))

similarity = dot_product / (magnitude1 * magnitude2)

print("Cosine similarity:", similarity)