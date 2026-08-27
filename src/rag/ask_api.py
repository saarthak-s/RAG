# File: ask_api.py
"""Minimal example client for the /ask endpoint."""

import requests

API_URL = "http://127.0.0.1:8000/ask"


def ask(query: str) -> str:
    response = requests.post(API_URL, json={"query": query})
    response.raise_for_status()
    return response.json()["answer"]


if __name__ == "__main__":
    question = "which decision tree model is used?"
    try:
        print("Answer from server:", ask(question))
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")