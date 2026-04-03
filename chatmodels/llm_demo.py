from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# simple message prompt
# result = model.invoke("What is the capital of India?")
# print(result)
# print(result.content[0]["text"])

# message prompt with roles and content
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of India?"},
]
response = model.invoke(conversation)
# print(response)
print(response.content[0]["text"])

