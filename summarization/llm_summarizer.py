from groq import Groq
from summarization.prompt import build_summary_prompt


client = Groq()


def summarize_with_llm(segments):
    combined_text = "\n".join(s["text"] for s in segments)

    prompt = build_summary_prompt(combined_text)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content
