import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

print("Tracing:",os.getenv("LANGSMITH_TRACING"))
print("API key:",os.getenv("LANGSMITH_API_KEY"))
print("Google API key:",os.getenv("GOOGLE_API_KEY"))

from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)