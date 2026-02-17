def build_tagging_prompt(segments):
    """
    segments: list of dicts with keys -> id, text
    """

    instructions = """
You are classifying classroom lecture transcript segments.

Allowed tags (choose exactly one per segment):
- LECTURE_CONTENT
- ADMINISTRATIVE
- OTHER_CHATTER

Definitions:

LECTURE_CONTENT:
ONLY sentences that directly teach subject knowledge,
including explanations, definitions, formulas, derivations,
worked examples, or conceptual reasoning.

ADMINISTRATIVE:
Any sentence talking about the lecture rather than teaching content.
Includes exam hints, importance statements, reminders,
instructions, pacing comments, syllabus discussion,
or guidance like "this is important" or "remember this".

OTHER_CHATTER:
Greetings, motivation, jokes, technical interruptions,
casual conversation, or non-academic speech.


Rules:
- One tag per segment
- Output format MUST be:
  segment_id: TAG
- One line per segment
- No explanations
"""

    body = "\n".join(
        f"{seg['id']}: {seg['text']}" for seg in segments
    )

    return instructions.strip() + "\n\n" + body
