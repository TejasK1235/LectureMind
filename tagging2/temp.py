import os
from groq import Groq

def main():
    # 1. Load API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not found in environment variables")

    # 2. Create client
    client = Groq(api_key=api_key)

    # 3. Simple test prompt
    prompt = "Is Batman better than Superman as a character? Give a short answer."

    # 4. Call the LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    # 5. Print result
    print("LLM response:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
