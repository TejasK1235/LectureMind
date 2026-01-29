# pipeline/run.py
import json
from pipeline.segment import segment_transcript
from pipeline.tagger import tag_segments


def run_pipeline(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_segments = json.load(f)

    segments = segment_transcript(raw_segments)
    tagged_segments = tag_segments(segments)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tagged_segments, f, indent=2)


if __name__ == "__main__":
    run_pipeline(
        input_path="data/raw/transcripts.json",
        output_path="data/tagged/tagged_output.json"
    )
