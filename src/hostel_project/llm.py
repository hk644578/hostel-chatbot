import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client=OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

Model="nvidia/nemotron-3-super-120b-a12b"

def generate_response(prompt):
    response=client.chat.completions.create(
        model=Model,
        messages=[
            {
                'role':"user",
                'content':prompt
            }
        ],
        temperature=0.2,
        max_tokens=500

    )
    return response.choices[0].message.content
