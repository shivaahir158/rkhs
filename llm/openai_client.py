"""
LLM client for heuristic synthesis (Step 5 in Algorithm 1).
Uses GPT-4 as specified in the paper.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment or .env file")
        _client = OpenAI(api_key=api_key)
    return _client


def query_llm(prompt, model="gpt-4", temperature=0.7):
    """
    Query the LLM to produce heuristic code (Algorithm 1, Step 9).
    """
    client = get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in high-level synthesis scheduling and "
                    "DAG optimization. You generate compact, deterministic Python "
                    "priority functions for resource-constrained list scheduling."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content
