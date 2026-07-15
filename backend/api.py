import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

RAG_PIPELINE = os.getenv(
    "RAG_PIPELINE",
    "legacy",
).strip().lower()

if RAG_PIPELINE in {
    "parent_child",
    "parent-child",
    "v2",
}:
    from rag_chat_parent_child import (
        rag_answer_parent_child as active_rag_answer,
    )

    ACTIVE_PIPELINE = "parent_child"

else:
    from rag_chat import (
        rag_answer as active_rag_answer,
    )

    ACTIVE_PIPELINE = "legacy"

print(
    f"FastAPI 当前 RAG 流程：{ACTIVE_PIPELINE}"
)


app = FastAPI(
    title="Medical RAG API"
)


class Question(BaseModel):
    question:str



@app.post("/ask")
def ask(
    q:Question
):

    result = active_rag_answer(
        q.question
    )

    return result
