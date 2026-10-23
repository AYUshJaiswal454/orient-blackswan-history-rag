"""
rag.py - Shared RAG pipeline for Orient BlackSwan History Bot.
Used by ask.py, server.py, and evaluate.py.
"""
import re
from typing import Dict, Any, List, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import chromadb

import config

# Global cache for reusable components
_VECTORSTORE: Optional[Chroma] = None
_LLM: Optional[ChatGoogleGenerativeAI] = None
_REWRITE_LLM: Optional[ChatGoogleGenerativeAI] = None

# In-memory session store: session_id -> list of {"role": "user"|"assistant", "content": str}
SESSION_HISTORIES: Dict[str, List[Dict[str, str]]] = {}


def get_vectorstore() -> Chroma:
    """Load or return the cached persistent Chroma vectorstore."""
    global _VECTORSTORE
    if _VECTORSTORE is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        chroma_client = chromadb.PersistentClient(path=str(config.DB_DIR))
        _VECTORSTORE = Chroma(
            client=chroma_client,
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
        )
    return _VECTORSTORE


def get_llm() -> ChatGoogleGenerativeAI:
    """Return configured Gemini chat model instance."""
    global _LLM
    if _LLM is None:
        _LLM = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=0,
        )
    return _LLM


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents with page numbers attached."""
    formatted = []
    for d in docs:
        page = d.metadata.get("page", "Unknown")
        formatted.append(f"[Page {page}] {d.page_content}")
    return "\n\n".join(formatted)


def normalize_text_for_match(text: str) -> str:
    """Normalize text by lowering case and stripping punctuation/whitespace for robust quote matching."""
    return re.sub(r"[\W_]+", "", text.lower())


def validate_evidence(raw_answer: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    """
    Validate that the model's answer is grounded in retrieved chunks.
    If 'Not found', unsupported, or citation is fabricated, returns 'Not found' with empty sources.
    """
    clean_answer = raw_answer.strip()
    lower_answer = clean_answer.lower()

    # Check for Not found indicators
    if (
        lower_answer == "not found"
        or lower_answer.startswith("not found")
        or ("not found" in lower_answer and len(clean_answer) < 40)
    ):
        return {"answer": "Not found", "sources": []}

    # Extract reference pattern: Reference: Page X - "quote"
    ref_match = re.search(
        r"Reference:\s*Page\s*(\d+)\s*[-–—:]\s*[\"']?(.*?)[\"']?$",
        clean_answer,
        re.MULTILINE | re.IGNORECASE,
    )

    if not ref_match:
        # Check alternative formatting: Page X: "quote" or Page X - "quote"
        ref_match = re.search(
            r"Page\s*(\d+)\s*[-–—:]\s*[\"']?(.*?)[\"']?$",
            clean_answer,
            re.MULTILINE | re.IGNORECASE,
        )

    if not ref_match:
        # No reference found when an answer was provided -> ungrounded
        return {"answer": "Not found", "sources": []}

    ref_page = int(ref_match.group(1))
    ref_quote = ref_match.group(2).strip().strip("\"'")

    # Validate that ref_quote is at most ~25 words (rule 2)
    quote_words = ref_quote.split()
    if len(quote_words) > 35:  # allowing small grace
        ref_quote = " ".join(quote_words[:25])

    normalized_quote = normalize_text_for_match(ref_quote)
    if not normalized_quote:
        return {"answer": "Not found", "sources": []}

    # Verify that ref_page exists in retrieved docs and quote is verbatim in that page's text
    quote_verified = False
    valid_doc = None

    for doc in retrieved_docs:
        doc_page = int(doc.metadata.get("page", 0))
        doc_norm = normalize_text_for_match(doc.page_content)
        # Check if page matches and normalized quote is a substring
        if doc_page == ref_page and normalized_quote in doc_norm:
            quote_verified = True
            valid_doc = doc
            break

    # If page number differed slightly but exact quote is in another retrieved page, adjust page
    if not quote_verified:
        for doc in retrieved_docs:
            doc_norm = normalize_text_for_match(doc.page_content)
            if normalized_quote in doc_norm:
                actual_page = int(doc.metadata.get("page", 0))
                # Update answer with the actual verified page
                clean_answer = re.sub(
                    r"Reference:\s*Page\s*\d+",
                    f"Reference: Page {actual_page}",
                    clean_answer,
                    flags=re.IGNORECASE,
                )
                quote_verified = True
                valid_doc = doc
                ref_page = actual_page
                break

    if not quote_verified:
        # Quote cannot be verified against retrieved chunks -> fail safe
        return {"answer": "Not found", "sources": []}

    # Build unique sources list
    sources = []
    seen_pages = set()
    # Put the validated document first
    if valid_doc is not None:
        p = int(valid_doc.metadata.get("page", ref_page))
        seen_pages.add(p)
        sources.append({
            "page": p,
            "snippet": valid_doc.page_content[:200].replace("\n", " ").strip(),
        })

    for doc in retrieved_docs:
        p = int(doc.metadata.get("page", 0))
        if p not in seen_pages and p > 0:
            seen_pages.add(p)
            sources.append({
                "page": p,
                "snippet": doc.page_content[:200].replace("\n", " ").strip(),
            })

    return {"answer": clean_answer, "sources": sources}


def rewrite_question_with_history(question: str, history: List[Dict[str, str]]) -> str:
    """Use Gemini to rephrase follow-up questions into standalone questions."""
    if not history:
        return question

    try:
        llm = get_llm()
        hist_str = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])
        prompt = (
            f"Given the conversation history below and a follow-up question, "
            f"rephrase the follow-up question into a single standalone search query.\n"
            f"If it is already standalone, return it as-is.\n"
            f"Do not answer the question, only output the standalone question.\n\n"
            f"History:\n{hist_str}\n\n"
            f"Follow-up question: {question}\n\nStandalone question:"
        )
        res = llm.invoke(prompt)
        text = res.content
        if isinstance(text, list):
            # Extract text if list of parts returned
            text = " ".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in text])
        standalone = str(text).strip().strip('"')
        return standalone if standalone else question
    except Exception:
        return question


def answer_question(question: str, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Primary interface for answering questions via LCEL RAG.
    Returns:
    {
        "answer": str,
        "sources": [{"page": int, "snippet": str}, ...]
    }
    """
    if not question or not question.strip():
        return {"answer": "Not found", "sources": []}

    # 1. Query reformulation using history if provided
    standalone_query = rewrite_question_with_history(question, history or [])

    # 2. Retrieve documents from persistent ChromaDB
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
    retrieved_docs = retriever.invoke(standalone_query)

    # 3. Format context
    context = format_docs(retrieved_docs)

    # 4. Construct LCEL chain
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", config.SYSTEM_PROMPT + "\n\nCONTEXT:\n{context}"),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 5. Generate answer
    try:
        raw_output = chain.invoke(standalone_query)
    except Exception as e:
        print(f"[!] Generation error: {e}")
        return {"answer": "Not found", "sources": []}

    # 6. Validate evidence and citation
    return validate_evidence(raw_output, retrieved_docs)


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get conversation history for a given session, keeping last 6 turns."""
    return SESSION_HISTORIES.setdefault(session_id, [])


def add_session_message(session_id: str, role: str, content: str):
    """Append a message to the session history, capping at 6 turns."""
    hist = get_session_history(session_id)
    hist.append({"role": role, "content": content})
    # Keep last 6 messages
    if len(hist) > 6:
        SESSION_HISTORIES[session_id] = hist[-6:]
