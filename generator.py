import os
import json
import requests
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv

load_dotenv()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, api_key=None):
    """
    Simple LLM text generator function
    
    Args:
        prompt: Text prompt to send to the LLM
        api_key: API key (defaults to LLM_API_KEY environment variable)
    
    Returns:
        Generated text response
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required. Provide as argument or set LLM_API_KEY environment variable")
    
    # This uses OpenAI's API format - adjust as needed
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"API error: {response.status_code} - {response.text}")
    
    return response.json()["choices"][0]["message"]["content"].strip()
