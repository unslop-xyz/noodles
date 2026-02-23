import os
import logging
from typing import Optional, Protocol
from openai import OpenAI, AsyncOpenAI
import google.generativeai as genai

logger = logging.getLogger(__name__)

class LLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        ...

    async def generate_async(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        ...

class OpenAIClient:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

    def generate(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}} if json_format else None
        )
        return response.output_text

    async def generate_async(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        response = await self.async_client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}} if json_format else None
        )
        return response.output_text

def _clean_json(text: str) -> str:
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```json"):
            text = "\n".join(lines[1:-1])
        elif lines[0].startswith("```"):
            text = "\n".join(lines[1:-1])
    return text.strip()

class GeminiClient:
    def __init__(self, model: str):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.model = genai.GenerativeModel(model)

    def generate(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        if json_format:
            full_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. Do not include any other text."
        
        response = self.model.generate_content(full_prompt)
        text = response.text
        if json_format:
            text = _clean_json(text)
        return text

    async def generate_async(self, system_prompt: str, user_prompt: str, json_format: bool = False) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        if json_format:
            full_prompt += "\n\nIMPORTANT: Return ONLY valid JSON. Do not include any other text."
        
        response = await self.model.generate_content_async(full_prompt)
        text = response.text
        if json_format:
            text = _clean_json(text)
        return text

def get_llm_client(model: Optional[str] = None) -> LLMClient:
    # Logic to select client based on environment or model name
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("UNSLOP_OPENAI_API_KEY")

    # If model looks like gemini, or only gemini key is available
    if (model and "gemini" in model.lower()) or (gemini_key and not openai_key):
        gemini_model = "gemini-2.0-flash"
        if model and "gemini" in model.lower():
             gemini_model = model
             # Ensure we don't pass OpenAI model names to Gemini
             if "gpt" in gemini_model.lower():
                 gemini_model = "gemini-2.0-flash"
        return GeminiClient(gemini_model)
    
    return OpenAIClient(model or "gpt-4.1")
