def build_summary_prompt(text: str) -> str:
    return f"""
You are an academic assistant.

Task:
Rewrite the following lecture excerpts into a concise, structured summary.

Rules:
- Do NOT add new information
- Do NOT remove important concepts
- Use clear headings and bullet points
- Keep the tone suitable for exam revision

Limit motivational or importance-related content to at most one short section.

Lecture excerpts:
{text}
""".strip()
