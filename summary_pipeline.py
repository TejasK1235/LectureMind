# pipeline.py
# Unified LectureMind pipeline

from tagging2.pipeline.run import run_tagging_pipeline
from summarization.run import run_summarization_pipeline


def run_lecturemind_pipeline(transcripts):
    """
    Full pipeline:
    raw transcripts → tagging → summarization
    """
    tagged_segments = run_tagging_pipeline(transcripts)
    summary = run_summarization_pipeline(tagged_segments)
    return summary


if __name__ == "__main__":
    import json

    with open("tagging2/data/raw/transcripts.json", "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    summary = run_lecturemind_pipeline(transcripts)


    print("\n===== FINAL SUMMARY =====\n")
    print(summary)
