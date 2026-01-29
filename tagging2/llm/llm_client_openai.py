# llm/llm_client.py
import os
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

        if self.api_key and OpenAI is not None:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def is_available(self) -> bool:
        print(self.client is not None)
        return self.client is not None

    # def classify(self, prompt: str) -> Optional[str]:
    #     if not self.client:
    #         return None

    #     try:
    #         response = self.client.responses.create(
    #             model="gpt-4.1-mini",
    #             input=prompt,
    #             max_output_tokens=10,
    #             temperature=0.0,
    #         )

    #         # Robust extraction: scan all output blocks
    #         for message in response.output:
    #             if "content" in message:
    #                 for block in message["content"]:
    #                     if block.get("type") == "output_text":
    #                         return block.get("text", "").strip()

    #         return None

    #     except Exception as e:
    #         return None

    def classify(self, prompt: str) -> Optional[str]:
        if not self.client:
            return None

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you have access
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            # Extract text from the response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content.strip()

            return None

        except Exception as e:
            print(f"LLM Error: {e}")  # Debug output
            return None
