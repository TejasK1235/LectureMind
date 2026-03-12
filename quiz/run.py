# quiz/run.py
# Entry point for both quiz types.
# 
# run_mcq_quiz()       — takes QB JSON output → list of MCQ dicts
# run_extempore_quiz() — takes concepts list  → list of topic dicts
#
# Called by api.py endpoints. Also runnable standalone via __main__.

import json
import os
from typing import List, Dict, Optional

from quiz.mcq_generator import generate_mcq
from quiz.extempore_generator import generate_extempore_topic
from quiz.prompts import MCQ_BLOOM_LEVELS

# Only concepts scoring above this get surfaced as extempore topics.
# Mirrors the QB pipeline's SCORE_THRESHOLD of 0.15 but slightly higher
# since we want only the meatiest concepts as presentation topics.
EXTEMPORE_SCORE_THRESHOLD = 0.25


def run_mcq_quiz(
    qb_result: Dict,
    concepts: List[Dict],
    num_questions: int = 10
) -> Dict:
    """
    Generates an MCQ quiz from QB output.

    Input:
        qb_result     — the full dict returned by run_qb_pipeline()
                        (has "questions" key with bloom-level buckets)
        concepts      — scored concept list from the same pipeline run
                        (needed to look up concept_text per question)
        num_questions — how many MCQs to attempt to generate

    Output:
        {
            "total":     10,
            "mcqs":      [{question, options, correct, correct_text}, ...],
            "warnings":  ["only 8 valid MCQs generated, requested 10"]
        }
    """

    # Step 1: collect candidate questions from allowed Bloom levels only
    candidates = []
    for level in MCQ_BLOOM_LEVELS:
        questions = qb_result.get("questions", {}).get(level, [])
        for q in questions:
            candidates.append({"question": q, "bloom": level})

    if not candidates:
        return {
            "total": 0,
            "mcqs": [],
            "warnings": [f"No questions found at Bloom levels: {MCQ_BLOOM_LEVELS}"]
        }

    # Step 2: build a lookup from concept text
    # We use the first concept's text as a fallback if we can't match —
    # not perfect but acceptable since all questions come from the same lecture
    concept_text_pool = " ".join(c["text"] for c in concepts) if concepts else ""

    # Step 3: generate MCQs up to num_questions
    mcqs = []
    warnings = []

    for candidate in candidates:
        if len(mcqs) >= num_questions:
            break

        result = generate_mcq(
            question=candidate["question"],
            concept_text=concept_text_pool
        )

        if result:
            result["bloom_level"] = candidate["bloom"]
            mcqs.append(result)
            print(f"[Quiz/MCQ] ✓ {candidate['bloom']} | {candidate['question'][:60]}...")
        else:
            print(f"[Quiz/MCQ] ✗ skipped: {candidate['question'][:60]}...")

    if len(mcqs) < num_questions:
        warnings.append(
            f"Only {len(mcqs)} valid MCQs generated, requested {num_questions}"
        )

    return {
        "total": len(mcqs),
        "mcqs": mcqs,
        "warnings": warnings
    }


def run_extempore_quiz(concepts: List[Dict]) -> Dict:
    """
    Generates a pool of presentation topics from scored concepts.

    Input:
        concepts — scored concept list from run_qb_pipeline()
                   (already sorted descending by score)

    Output:
        {
            "total":  8,
            "topics": [
                {"concept_id": "...", "title": "...", "score": 0.84, "word_count": 210},
                ...
            ],
            "warnings": []
        }
    """

    # Filter to only concepts substantial enough for a presentation
    eligible = [c for c in concepts if c["score"] >= EXTEMPORE_SCORE_THRESHOLD]

    if not eligible:
        # Fallback: take top half regardless of threshold
        eligible = concepts[:max(1, len(concepts) // 2)]

    topics = []
    warnings = []

    for concept in eligible:
        result = generate_extempore_topic(concept)
        if result:
            topics.append(result)
            print(f"[Quiz/Extempore] ✓ {result['title']}")
        else:
            print(f"[Quiz/Extempore] ✗ skipped concept {concept['concept_id']}")

    if not topics:
        warnings.append("No extempore topics could be generated.")

    # Keep sorted by score descending so frontend can slice top-N easily
    topics.sort(key=lambda t: t["score"], reverse=True)

    return {
        "total": len(topics),
        "topics": topics,
        "warnings": warnings
    }

# ── Standalone test runner ────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Configure your test run here ─────────────────────────────────────────
    NUM_MCQS = 10         # how many MCQs to generate
    NUM_TOPICS = 5         # how many extempore topics to return (top N by score)
    # ─────────────────────────────────────────────────────────────────────────

    # Load real QB output (which now includes concepts after your pipeline update)
    qb_path = os.path.join(os.path.dirname(__file__), "..", "qb_output.json")

    with open(qb_path, "r", encoding="utf-8") as f:
        qb_result = json.load(f)

    concepts = qb_result.get("concepts", [])

    if not concepts:
        print("ERROR: No concepts found in QB output. Make sure you ran the updated run_qb_pipeline().")
        exit(1)

    print(f"\n[Quiz] Loaded {len(concepts)} concepts from QB output.")
    print(f"[Quiz] Generating {NUM_MCQS} MCQs and {NUM_TOPICS} extempore topics...\n")

    # ── MCQ Quiz ──────────────────────────────────────────────────────────────
    print("===== MCQ QUIZ =====\n")
    mcq_result = run_mcq_quiz(qb_result, concepts, num_questions=NUM_MCQS)

    for i, mcq in enumerate(mcq_result["mcqs"], 1):
        print(f"Q{i}. [{mcq['bloom_level']}] {mcq['question']}")
        for letter, text in mcq["options"].items():
            marker = " ✓" if letter == mcq["correct"] else ""
            print(f"   {letter}. {text}{marker}")
        print()

    if mcq_result["warnings"]:
        print("Warnings:", mcq_result["warnings"])

    # ── Extempore Quiz ────────────────────────────────────────────────────────
    print("\n===== EXTEMPORE TOPICS =====\n")
    extempore_result = run_extempore_quiz(concepts)

    # Slice to NUM_TOPICS — full list is generated, frontend/teacher picks from top N
    top_topics = extempore_result["topics"][:NUM_TOPICS]

    for i, topic in enumerate(top_topics, 1):
        print(f"{i}. {topic['title']}")
        print(f"   score: {topic['score']} | words: {topic['word_count']}")
        print()

    if extempore_result["warnings"]:
        print("Warnings:", extempore_result["warnings"])

    # ── Save full output ──────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "..", "quiz_output_sample.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "mcq_quiz": {**mcq_result},
            "extempore_quiz": {
                "total": len(top_topics),
                "topics": top_topics,
                "warnings": extempore_result["warnings"]
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"Full output saved to quiz_output_sample.json")

# if __name__ == "__main__":

#     # Load QB output and tagged segments
#     qb_path = os.path.join(os.path.dirname(__file__), "..", "qb_output_sample.json")

#     with open(qb_path, "r", encoding="utf-8") as f:
#         qb_result = json.load(f)

#     # For standalone testing we use a dummy concept list
#     # In real usage concepts come from run_qb_pipeline()
#     dummy_concepts = [
#         {
#             "concept_id": "concept_test_01",
#             "text": "The inode stores all metadata about a file including permissions, "
#                     "timestamps, and pointers to data blocks. Every file in Linux has "
#                     "exactly one inode. The inode number is used to locate the inode "
#                     "on disk. Directories map filenames to inode numbers.",
#             "score": 0.85,
#             "word_count": 45,
#             "emphasis_count": 2
#         }
#     ]

#     print("\n===== MCQ QUIZ =====\n")
#     mcq_result = run_mcq_quiz(qb_result, dummy_concepts, num_questions=5)
#     for i, mcq in enumerate(mcq_result["mcqs"], 1):
#         print(f"Q{i}. {mcq['question']}")
#         for letter, text in mcq["options"].items():
#             marker = " ✓" if letter == mcq["correct"] else ""
#             print(f"   {letter}. {text}{marker}")
#         print()

#     print("\n===== EXTEMPORE TOPICS =====\n")
#     extempore_result = run_extempore_quiz(dummy_concepts)
#     for i, topic in enumerate(extempore_result["topics"], 1):
#         print(f"{i}. {topic['title']}  (score: {topic['score']})")

#     # Save outputs
#     out_path = os.path.join(os.path.dirname(__file__), "..", "quiz_output_sample.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump({
#             "mcq_quiz": mcq_result,
#             "extempore_quiz": extempore_result
#         }, f, indent=2, ensure_ascii=False)

#     print(f"\nFull output saved to quiz_output_sample.json")