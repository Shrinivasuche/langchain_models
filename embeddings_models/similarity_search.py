import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview", output_dimensionality=768
)

documents = [
    "New Delhi is the seat of the Government of India and the capital city.",
    "The Taj Mahal is a white marble mausoleum in the city of Agra.",
    "The Indian Premier League (IPL) is a professional T20 cricket league.",
    "The President of India resides in the Rashtrapati Bhavan in Delhi.",
    "Software engineering is the systematic application of engineering principles to software."
]

document_vector = embeddings_model.embed_documents(documents)

query = "Where is the capital of India located?"
query_vector = embeddings_model.embed_query(query)

scores = cosine_similarity([query_vector], document_vector)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"Query: {query}\n")
print(documents[index])
print(f"Similarity Score is : {score}")
