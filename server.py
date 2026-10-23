"""
server.py - FastAPI backend for Orient BlackSwan History Bot.
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm Vectorstore and LLM on application startup."""
    print("[*] Server startup: Preloading Chroma vectorstore and embedding models...")
    try:
        rag.get_vectorstore()
        rag.get_llm()
        print("[*] Vectorstore and LLM successfully preloaded.")
    except Exception as e:
        print(f"[!] Warning during model warm-up: {e}", file=sys.stderr)
    yield
    print("[*] Server shutdown clean.")


app = FastAPI(
    title="Orient BlackSwan History Bot API",
    description="RAG Question-Answering API for Orient BlackSwan History Catalogue 2024-25",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
# Security note: Permissive wildcard CORS is enabled for local evaluation & frontend development.
# In a production environment, restrict allow_origins to explicit domain origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question about the catalogue")
    session_id: Optional[str] = Field("default", description="Conversation session ID")


class SourceItem(BaseModel):
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question_endpoint(req: AskRequest):
    """
    RAG question-answering endpoint with session memory.
    """
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = req.session_id.strip() if req.session_id and req.session_id.strip() else "default"

    try:
        # Retrieve session conversation history (capped at 6 turns)
        history = rag.get_session_history(session_id)
        result = rag.answer_question(q, history=history)

        answer = result.get("answer", "Not found")
        sources = result.get("sources", [])

        # Update session memory
        rag.add_session_message(session_id, "user", q)
        rag.add_session_message(session_id, "assistant", answer)

        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        # Never expose stack traces or environment variables to client
        print(f"[!] Server error answering question: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your query.",
        )


@app.get("/")
def serve_index():
    """Serve the single-file chat web UI."""
    index_path = config.BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Orient BlackSwan History Bot API is running. index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
