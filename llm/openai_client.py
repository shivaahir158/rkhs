import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def ask_llm(prompt):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in DAG scheduling and heterogeneous computing."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content