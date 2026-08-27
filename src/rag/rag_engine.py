# File: rag_engine.py

import os

import chromadb
from dotenv import load_dotenv

load_dotenv()


def build_llm_client():
    """LLM_PROVIDER=ollama routes to a local model with no rate limit at all
    — use this for iterative dev. LLM_PROVIDER=groq (default) uses the
    hosted API for final testing."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        from openai import OpenAI
        model = os.getenv("LLM_MODEL", "llama3.1:8b")
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        return client, model

    from groq import Groq
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    print(f"[INFO] Using Groq API with model {model}.")
    client = Groq()
    return client, model


class CodebaseAssistant:
    def __init__(self):
        print("[INFO] Booting up CodebaseAssistant Engine...")

        self.client_db = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.client_db.get_collection(name="dt_api_codebase")
        except Exception as e:
            raise RuntimeError(
                "Vector DB not found or empty. Run `python build_db.py` first."
            ) from e

        self.llm_client, self.model = build_llm_client()

        print(f"[INFO] Engine ready. Provider={os.getenv('LLM_PROVIDER', 'groq')} "
              f"Model={self.model} Chunks={self.collection.count()}")

    def _retrieve(self, query: str):
        results = self.collection.query(query_texts=[query], n_results=5)
        return results["documents"][0], results["metadatas"][0]

    def _build_prompt(self, query: str, chunks: list, metadata: list) -> str:
        context_blocks = [
            f"Source: {m['file']} -> {m['name']}\nCode:\n{c}"
            for c, m in zip(chunks, metadata)
        ]
        context_string = "\n\n---\n\n".join(context_blocks)
        return f"""You are an expert engineering assistant. Answer the user's question using ONLY the following codebase context.
If the answer is not contained in the context, strictly reply: "I do not have enough context to answer that."

Context:
{context_string}

Question: {query}"""

    def _call_llm(self, prompt: str) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def ask(self, query: str) -> dict:
        chunks, metadata = self._retrieve(query)
        prompt = self._build_prompt(query, chunks, metadata)

        try:
            answer_text = self._call_llm(prompt)
        except Exception as e:
            answer_text = f"API Error: {str(e)}"

        unique_sources = list({f"{m['file']} ({m['name']})" for m in metadata})
        return {"answer": answer_text, "sources": unique_sources}