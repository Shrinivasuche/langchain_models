from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", output_dimensionality=768
)

# vector = embeddings_model.embed_query("hello, world!")
# print(vector[:5])


# batch embeddings
batch_vector = embeddings_model.embed_documents(["hello, world!", "goodbye, world!"])
print(batch_vector[0][:5])
