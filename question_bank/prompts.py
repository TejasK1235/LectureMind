# question_bank/prompts.py

from typing import Optional

BLOOM_DESCRIPTIONS = {
    "remember":   "recall facts, definitions, and basic concepts directly stated in the text",
    "understand": "explain ideas or concepts in your own words, interpret meaning",
    "apply":      "use the concept in a new or practical situation",
    "analyze":    "break down the concept, examine relationships, identify causes or structure",
    "evaluate":   "make a judgment, critique, or justify a position based on the concept",
    "create":     "propose, design, or formulate something new based on the concept"
}

# BLOOM_STARTERS = {
#     "remember":   ["What is...", "Define...", "List...", "State...", "Name..."],
#     "understand": ["Explain why...", "Describe how...", "What does ... mean?", "Summarize..."],
#     "apply":      ["How would you use ... in a situation where...", "Given that ..., calculate/determine...", "Illustrate how ... applies when..."],
#     "analyze":    ["What are the differences between ... and ...?", "Why does ... cause ...?", "Break down the relationship between...", "What are the implications of..."],
#     "evaluate":   ["To what extent is ... true?", "Assess the validity of...", "Critique the claim that...", "Is ... an appropriate approach? Why?", "Justify why..."],
#     "create":     ["Design a ... that...", "Propose a method to...", "Formulate a hypothesis about...", "How would you construct..."]
# }
BLOOM_STARTERS = {
    "remember": [
        "What is...",
        "Define...",
        "List...",
        "State...",
        "Name...",
        "Which of the following describes...",
        "What does ... refer to?",
        "Identify...",
        "When does ... occur?",
        "What are the steps involved in...?",
    ],
    "understand": [
        "Explain why...",
        "Describe how...",
        "What does ... mean?",
        "Summarize...",
        "In your own words, what is...?",
        "How would you interpret...?",
        "What is the relationship between ... and ...?",
        "Why is ... important in this context?",
        "How does ... work?",
        "What is the significance of...?",
    ],
    "apply": [
        "How would you use ... in a situation where...?",
        "Given that ..., how would you determine...?",
        "Illustrate how ... applies when...",
        "If you were to implement ..., what steps would you follow?",
        "How could ... be used to solve...?",
        "Demonstrate how ... would behave when...",
        "What would happen if ... were applied to...?",
        "Using the concept of ..., how would you approach...?",
        "How would you modify ... to achieve...?",
        "In what situation would ... be most useful?",
    ],
    "analyze": [
        "What are the differences between ... and ...?",
        "Why does ... cause ...?",
        "Break down the relationship between...",
        "What are the implications of...?",
        "How does ... affect ...?",
        "What factors contribute to...?",
        "Compare and contrast ... and ...",
        "Why is ... structured the way it is?",
        "What would change if ... were removed?",
        "How does ... depend on ...?",
    ],
    "evaluate": [
        "To what extent is ... true?",
        "Assess the validity of...",
        "Critique the claim that...",
        "Is ... an appropriate approach? Why?",
        "Justify why...",
        "What are the strengths and weaknesses of...?",
        "Would you consider ... an effective solution? Why or why not?",
        "How would you judge the effectiveness of...?",
        "Is ... always the best choice? Under what conditions might it fail?",
        "What evidence supports or contradicts the idea that...?",
    ],
    "create": [
        "Design a ... that...",
        "Propose a method to...",
        "Formulate a hypothesis about...",
        "How would you construct...?",
        "What approach would you take to build...?",
        "Suggest an improvement to ... that addresses...",
        "How would you redesign ... to better handle...?",
        "Develop a strategy for...",
        "What would a new system for ... look like?",
        "Propose an alternative to ... that solves...",
    ],
}

MAX_CONCEPT_WORDS = 200
MAX_SLIDE_WORDS = 150    # slide text is usually denser so slightly shorter cap


def build_question_prompt(
    concept_text: str,
    bloom_level: str,
    num_questions: int,
    slide_text: Optional[str] = None    # NEW parameter, None = no slide content
) -> str:

    bloom_desc = BLOOM_DESCRIPTIONS.get(bloom_level, "think critically about the concept")
    starters = BLOOM_STARTERS.get(bloom_level, [])
    starters_text = ", ".join(f'"{s}"' for s in starters)

    # Trim transcript text
    words = concept_text.split()
    if len(words) > MAX_CONCEPT_WORDS:
        concept_text = " ".join(words[:MAX_CONCEPT_WORDS]) + "..."

    # Build the content section depending on whether slide text is available
    if slide_text:
        # Trim slide text
        slide_words = slide_text.split()
        if len(slide_words) > MAX_SLIDE_WORDS:
            slide_text = " ".join(slide_words[:MAX_SLIDE_WORDS]) + "..."

        content_section = f"""Lecture Transcript (what the instructor said):
{concept_text}

Slide Content (what was shown on screen during this section):
{slide_text}

Generate questions that draw from both sources above where relevant."""

    else:
        content_section = f"""Lecture Content:
{concept_text}"""

    return f"""You are an academic question generator for university students.

Your task is to generate exactly {num_questions} exam question(s) based ONLY on the content provided below.

Bloom's Taxonomy Level: {bloom_level}
What this means: Questions must require students to {bloom_desc}.

Use the following as cognitive guides to calibrate the correct Bloom level: {starters_text}
These are thinking anchors, NOT required prefixes. Each question must read like a natural exam question — varied in structure and phrasing. No two questions should open with the same word or phrase.
Each question must use a different sentence structure.

Avoid repeating patterns such as:
- "How would you..."
- "Suppose you..."
- "To what extent..."

Vary across:
- direct conceptual questions
- scenario-based questions
- comparison questions
- cause-effect reasoning

If two questions feel structurally similar, rewrite one.

Rules:
- Use ONLY information present in the provided content. Do not add outside knowledge.
- Each question must genuinely be at the {bloom_level} level — not easier, not harder.
- Questions must be descriptive or short-answer style. No MCQs.
- Each question must be a complete, clearly worded sentence.
- Do NOT start multiple questions with the same word or phrase.
- Output ONLY a JSON array of strings. No explanation, no numbering, no extra text.

Example output format:
["Question one here?", "Question two here?"]

{content_section}

JSON Output:"""