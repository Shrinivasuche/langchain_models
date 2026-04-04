from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

text = "Delhi is the capital of India"

# 3. Convert text to a vector (List of floats)
vector = embeddings.embed_query(text)

print(f"Vector Length: {len(vector)}")
print(f"First 5 dimensions: {vector[:5]}")
