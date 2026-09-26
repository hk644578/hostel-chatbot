import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

print("API key loaded:", api_key is not None)
print("API key length:", len(api_key) if api_key else 0)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
)

response = client.embeddings.create(
    model="nvidia/nemotron-3-embed-1b",
    input="What are the hostel rules at NIT Jalandhar?",
    extra_body={
        "input_type": "query"
    }
)

print("Success!")
print("Dimension:", len(response.data[0].embedding))