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
Academic explanations, definitions, concepts, examples, derivations.

ADMINISTRATIVE:
Exams, marks, assignments, syllabus, deadlines.

OTHER_CHATTER:
Greetings, casual talk, jokes, classroom management, attendance.

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
