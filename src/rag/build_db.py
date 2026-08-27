# File: build_db.py

import ast
import re
import requests
import chromadb

REPO_OWNER = "saarthak-s"
REPO_NAME = "dt-load-predictor"
BASE_URL = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/main/"

REPO_FILES = [
    ".gitignore",
    "Dockerfile",
    "README.md",
    "main.py",
    "pyproject.toml",
    "requirements.txt",
    "test_main.py",
    "train_dt_forecast_model.py",
]


def extract_chunks(filename: str, content: str) -> list:
    """Routes the file to the correct parsing strategy based on its extension."""
    extracted = []

    if filename.endswith(".py"):
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    chunk_text = ast.get_source_segment(content, node)
                    extracted.append({
                        "metadata": {"name": node.name, "type": "Python", "file": filename},
                        "content": chunk_text,
                    })
        except SyntaxError:
            print(f"[ERROR] Syntax error parsing {filename}")

    elif filename.endswith(".md"):
        sections = re.split(r"(?m)^#{1,3}\s", content)
        for section in sections:
            if section.strip():
                title = section.strip().split("\n")[0][:30]
                extracted.append({
                    "metadata": {"name": f"Section: {title}", "type": "Markdown", "file": filename},
                    "content": section.strip(),
                })

    elif filename in ["Dockerfile", "pyproject.toml", "requirements.txt", ".gitignore"]:
        extracted.append({
            "metadata": {"name": "Entire File", "type": "Config", "file": filename},
            "content": content,
        })

    return extracted


def fetch_and_chunk_repo() -> list:
    chunks = []
    print("--- PHASE 1: Fetching Repository ---")
    for file in REPO_FILES:
        response = requests.get(BASE_URL + file)
        if response.status_code == 200:
            file_chunks = extract_chunks(file, response.text)
            chunks.extend(file_chunks)
            print(f"[SUCCESS] {file} -> Extracted {len(file_chunks)} chunks.")
        else:
            print(f"[ERROR] Failed to fetch {file}. Status: {response.status_code}")
    print(f"\nTotal chunks ready for the database: {len(chunks)}\n")
    return chunks


def build_vector_store(chunks: list):
    print("--- PHASE 2: Updating ChromaDB ---")
    client_db = chromadb.PersistentClient(path="./chroma_db")
    collection = client_db.get_or_create_collection(name="dt_api_codebase")

    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Upsert overwrites old chunks by ID instead of duplicating on re-run
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[COMPLETE] Safely embedded {collection.count()} total chunks into the database.")


def main():
    chunks = fetch_and_chunk_repo()
    build_vector_store(chunks)


if __name__ == "__main__":
    main()