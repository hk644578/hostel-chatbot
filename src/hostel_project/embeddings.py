import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)
def get_embeddings(texts):
    response = client.embeddings.create(
        model="nvidia/nemotron-3-embed-1b",
        input=texts
    )
    return [item.embedding for item in response.data]