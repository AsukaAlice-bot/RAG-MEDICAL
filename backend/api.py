from fastapi import FastAPI
from pydantic import BaseModel

from rag_chat import rag_answer


app = FastAPI(
    title="Medical RAG API"
)


class Question(BaseModel):
    question:str



@app.post("/ask")
def ask(
    q:Question
):

    result = rag_answer(
        q.question
    )

    return result