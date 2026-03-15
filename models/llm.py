from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


def get_chatgroq_model():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
