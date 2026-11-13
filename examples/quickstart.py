"""Engram quickstart in Python — store memories for a user, then ask about them.

    pip install requests
    ENGRAM_API_KEY=your_key python quickstart.py
"""

import os

import requests

API     = "https://engram-c5m6.onrender.com/v1"
HEADERS = {"X-API-Key": os.environ["ENGRAM_API_KEY"]}
USER_ID = "demo-user"


def remember(text: str) -> None:
    requests.post(f"{API}/ingest", headers=HEADERS, json={"user_id": USER_ID, "content": text}, timeout=90).raise_for_status()
    print("remembered:", text)


def ask(question: str, top_k: int = 3) -> dict:
    response = requests.post(f"{API}/query", headers=HEADERS,
                             json={"user_id": USER_ID, "query": question, "top_k": top_k}, timeout=90)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    remember("I am training for a marathon in April")
    remember("My left knee hurts after long runs")
    remember("I switched to running in the mornings because evenings were too busy")
    print()
    print(ask("What should I keep in mind when planning this user's training?"))
