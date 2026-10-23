#!/usr/bin/env python3
"""
ingest.py - Ingest Orient BlackSwan History Catalogue 2024-25 PDF into ChromaDB.
"""
import os
import sys
import time
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb

import config


def clean_text(text: str) -> str:
    """
    Clean repeated whitespace and blank lines while preserving tabular
    entries, prices, dates, and headings.
    """
    if not text:
        return ""
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Clean multiple spaces/tabs per line without destroying newlines
    lines = []
    for line in text.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(cleaned_line)
    joined = "\n".join(lines)
    # Collapse 3 or more consecutive newlines into 2
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def run_ingestion():
    start_time = time.time()
    pdf_path = config.PDF_PATH

    print(f"[*] Starting ingestion for PDF: {pdf_path}")
    if not pdf_path.exists():
        print(f"[!] Error: PDF file not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load document page-by-page using PyPDFLoader
    try:
        loader = PyPDFLoader(str(pdf_path))
        raw_pages = loader.load()
    except Exception as e:
        print(f"[!] Error loading PDF with PyPDFLoader: {e}", file=sys.stderr)
        sys.exit(1)

    total_pages_loaded = len(raw_pages)
    print(f"[*] Total raw pages loaded: {total_pages_loaded}")

    cleaned_docs = []
    empty_pages_skipped = 0

    for idx, page_doc in enumerate(raw_pages):
        raw_content = page_doc.page_content or ""
        cleaned_content = clean_text(raw_content)

        # Human-readable 1-based page number
        page_num = page_doc.metadata.get("page", idx) + 1

        if not cleaned_content or len(cleaned_content) < 5:
            empty_pages_skipped += 1
            continue

        cleaned_docs.append(
            Document(
                page_content=cleaned_content,
                metadata={
                    "page": int(page_num),
                    "source": pdf_path.name,
                },
            )
        )

    print(f"[*] Valid pages with text: {len(cleaned_docs)}")
    print(f"[*] Empty pages skipped: {empty_pages_skipped}")

    if not cleaned_docs:
        print("[!] Error: No extractable text found across all pages.", file=sys.stderr)
        sys.exit(1)

    # 2. Chunking with RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(cleaned_docs)
    total_chunks = len(chunks)
    print(f"[*] Total text chunks created: {total_chunks}")

    # Ensure metadata integrity on chunks
    for chunk in chunks:
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = 1
        chunk.metadata["page"] = int(chunk.metadata["page"])
        chunk.metadata["source"] = pdf_path.name

    # 3. Initialize local HuggingFace embeddings
    print(f"[*] Initializing local embeddings: {config.EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print(f"[!] Error initializing embeddings: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Setup ChromaDB persistent storage and reset existing collection
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(config.DB_DIR))

    try:
        existing_collections = [c.name for c in chroma_client.list_collections()]
        if config.CHROMA_COLLECTION_NAME in existing_collections:
            print(f"[*] Resetting previous collection: '{config.CHROMA_COLLECTION_NAME}'")
            chroma_client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
    except Exception as e:
        print(f"[*] Notice when inspecting existing collections: {e}")

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # 5. Insert in batches of 200 chunks
    batch_size = 200
    print(f"[*] Inserting {total_chunks} chunks into ChromaDB in batches of {batch_size}...")
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"    - Inserted batch {i // batch_size + 1} ({len(batch)} chunks)")

    elapsed_time = round(time.time() - start_time, 2)
    print("\n" + "=" * 50)
    print("INGESTION SUMMARY:")
    print(f"  Total pages loaded:    {total_pages_loaded}")
    print(f"  Empty pages skipped:   {empty_pages_skipped}")
    print(f"  Text chunks created:   {total_chunks}")
    print(f"  Chroma collection:     {config.CHROMA_COLLECTION_NAME}")
    print(f"  Storage directory:     {config.DB_DIR}")
    print(f"  Total time taken:      {elapsed_time}s")
    print("=" * 50)


if __name__ == "__main__":
    run_ingestion()
