from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

from dotenv import load_dotenv

load_dotenv()

model = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.7,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
chat = ChatHuggingFace(llm=model)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of India?"},
]

try:
    response = chat.invoke(conversation)
    print("\n--- AI Response ---")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
