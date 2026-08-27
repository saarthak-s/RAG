# File: rag_engine.py

import hashlib
import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()


class DiskCache:
    """Caches LLM responses on disk keyed by the exact prompt sent.
    Repeating the same question during development costs zero API calls
    after the first time."""

    def __init__(self, cache_dir: str = ".llm_cache"):
        self.dir = Path(cache_dir)
        self.dir.mkdir(exist_ok=True)

    def _key(self, prompt: str) -> Path:
        h = hashlib.sha256(prompt.encode()).hexdigest()
        return self.dir / f"{h}.json"

    def get(self, prompt: str):
        path = self._key(prompt)
        if path.exists():
            return json.loads(path.read_text())["response"]
        return None

    def set(self, prompt: str, response: str):
        self._key(prompt).write_text(json.dumps({"response": response}))


class SyncRateLimiter:
    """A rate limiter that persists to disk and calibrates via API headers,
    tracking both Requests Per Minute (RPM) and Tokens Per Minute (TPM)."""

    def __init__(self, state_file: str = ".ratelimit.json", rpm: int = 30, tpm: int = 80000):
        self.file = Path(state_file)
        self.rpm = rpm
        self.tpm = tpm
        
        # Create the file with max limits if it doesn't exist
        if not self.file.exists():
            self._save({
                "rem_req": self.rpm, 
                "rem_tok": self.tpm, 
                "reset_req": time.time(), 
                "reset_tok": time.time()
            })

    def _save(self, state: dict):
        self.file.write_text(json.dumps(state))

    def _load(self) -> dict:
        try:
            return json.loads(self.file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {
                "rem_req": self.rpm, 
                "rem_tok": self.tpm, 
                "reset_req": time.time(), 
                "reset_tok": time.time()
            }

    def wait_if_needed(self, estimated_tokens: int):
        """Checks the local JSON file before making a request and sleeps if limits are hit."""
        state = self._load()
        now = time.time()
        
        # 1. Check Requests Per Minute (RPM)
        if state["rem_req"] <= 0 and now < state["reset_req"]:
            sleep_time = state["reset_req"] - now + 0.1
            print(f"[RateLimiter] 30 RPM limit reached. Sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            now = time.time()
            state["rem_req"], state["rem_tok"] = self.rpm, self.tpm
            
        # 2. Check Tokens Per Minute (TPM)
        if state["rem_tok"] < estimated_tokens and now < state["reset_tok"]:
            sleep_time = state["reset_tok"] - now + 0.1
            print(f"[RateLimiter] 80k TPM limit reached. Sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
            state["rem_req"], state["rem_tok"] = self.rpm, self.tpm

        # Optimistically deduct usage locally before the API responds.
        # This prevents rapid concurrent queries from bypassing the limits.
        state["rem_req"] -= 1
        state["rem_tok"] -= estimated_tokens
        self._save(state)

    def sync_from_headers(self, headers):
        """Updates the local JSON file using the API's exact truth."""
        state = self._load()
        
        # Extract the ground truth directly from Groq/OpenAI headers
        rem_req = headers.get("x-ratelimit-remaining-requests")
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        
        # Sync requests
        if rem_req is not None:
            state["rem_req"] = int(rem_req)
            if state["reset_req"] <= time.time():
                 state["reset_req"] = time.time() + 60
                 
        # Sync tokens
        if rem_tok is not None:
            state["rem_tok"] = int(rem_tok)
            if state["reset_tok"] <= time.time():
                 state["reset_tok"] = time.time() + 60

        self._save(state)


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
        self.cache = DiskCache()
        
        # Initialize synchronized limiter with specified hard limits
        rpm = int(os.getenv("RATE_LIMIT_RPM", "30"))
        tpm = int(os.getenv("RATE_LIMIT_TPM", "80000"))
        self.limiter = SyncRateLimiter(rpm=rpm, tpm=tpm)

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

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
    def _call_llm(self, prompt: str) -> str:
        # Estimate token usage: roughly 4 characters per token + max completion buffer
        estimated_tokens = (len(prompt) // 4) + 1024
        
        # Wait before hitting the API if limits are breached
        self.limiter.wait_if_needed(estimated_tokens)

        # Check if the client supports raw response access (Groq/OpenAI APIs)
        if hasattr(self.llm_client.chat.completions, "with_raw_response"):
            raw_response = self.llm_client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            # Sync ground truth from headers
            self.limiter.sync_from_headers(raw_response.headers)
            parsed_response = raw_response.parse()
            return parsed_response.choices[0].message.content
        else:
            # Fallback for mock clients or local Ollama instances
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

    def ask(self, query: str) -> dict:
        chunks, metadata = self._retrieve(query)
        prompt = self._build_prompt(query, chunks, metadata)

        cached = self.cache.get(prompt)
        if cached is not None:
            answer_text = cached
        else:
            try:
                answer_text = self._call_llm(prompt)
                self.cache.set(prompt, answer_text)
            except Exception as e:
                answer_text = f"API Error after retries: {str(e)}"

        unique_sources = list({f"{m['file']} ({m['name']})" for m in metadata})
        return {"answer": answer_text, "sources": unique_sources}