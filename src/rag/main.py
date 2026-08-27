# File: main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag_engine import CodebaseAssistant

app = FastAPI(
    title="DT Forecast Codebase Assistant",
    description="An AI-powered API that answers questions about its own architecture.",
)

rag_assistant = CodebaseAssistant()


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=AskResponse)
async def ask_codebase(request: AskRequest):
    """Takes a developer's question and returns an LLM-generated answer
    grounded strictly in the repository's source code."""
    try:
        return rag_assistant.ask(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))