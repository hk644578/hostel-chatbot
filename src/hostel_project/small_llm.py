import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
model="nvidia/nemotron-3.5-lightning-30b-a3b"

client=OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

def generate_response(prompt):
    response=client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=500,
        messages=[
            {'role':'user','content':prompt}
        ]
    )
    return response.choices[0].message.content