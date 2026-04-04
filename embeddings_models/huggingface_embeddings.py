from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

# single embedding
text = "Delhi is the capital of India"
vector = embeddings.embed_query(text)

print(f"Vector Length: {len(vector)}")
print(f"First 5 dimensions: {vector[:5]}")

# batch embeddings
batch_texts = ["Delhi is the capital of India", "Paris is the capital of France"]
batch_vectors = embeddings.embed_documents(batch_texts)
print(f"Batch Vector Length: {len(batch_vectors[0])}")
print(f"First 5 dimensions of first vector: {batch_vectors[0][:5]} \n\n\n")
print("Batch Vectors: \n", batch_vectors)

