# question_bank/question_generator.py

import json
import time
import os
from typing import List, Optional
from groq import Groq
from tagging2.schema import Concept
from tagging2.config import GROQ_MODEL_QB
from question_bank.prompts import build_question_prompt

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MAX_RETRIES = 2


def generate_questions_for_concept(
    concept: Concept,
    bloom_level: str,
    num_questions: int
) -> List[str]:

    if num_questions <= 0:
        return []

    # Pass slide_text only if the concept has one attached
    slide_text: Optional[str] = concept.get("slide_text", None)

    prompt = build_question_prompt(
        concept_text=concept["text"],
        bloom_level=bloom_level,
        num_questions=num_questions,
        slide_text=slide_text          # NEW — None if no slide matched
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL_QB,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            raw = response.choices[0].message.content.strip()
            questions = _parse_response(raw)
            if questions:
                return questions

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                wait = 10 * (attempt + 1)
                print(f"[QB] Rate limit hit, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                print(f"[QB] LLM call failed on attempt {attempt + 1}: {e}")

    print(f"[QB] Warning: no questions generated for concept {concept['concept_id']} at {bloom_level}")
    return []


def _parse_response(raw: str) -> List[str]:
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [q.strip() for q in parsed if isinstance(q, str) and q.strip()]
    except json.JSONDecodeError:
        pass

    return []