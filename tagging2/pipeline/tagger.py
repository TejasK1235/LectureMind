from typing import List

from tagging2.schema import TranscriptSegment, TaggedSegment
from tagging2.rules.rule_based import apply_rules
from tagging2.llm.classify import classify_batch


BATCH_SIZE = 10


def tag_segments(segments: List[TranscriptSegment]) -> List[TaggedSegment]:
    # Step 1: apply rules
    tagged: List[TaggedSegment] = []

    for seg in segments:
        rule_tag = apply_rules(seg["text"])
        tagged.append({
            **seg,
            "tag": rule_tag if rule_tag else "UNDECIDED"
        })

    # Step 2: resolve UNDECIDED in batches
    buffer = []

    for seg in tagged:
        if seg["tag"] == "UNDECIDED":
            buffer.append(seg)
        else:
            _resolve_buffer(buffer)
            buffer.clear()

    _resolve_buffer(buffer)

    return tagged


def _resolve_buffer(buffer):
    if not buffer:
        return

    for i in range(0, len(buffer), BATCH_SIZE):
        batch = buffer[i:i + BATCH_SIZE]
        results = classify_batch(batch)

        for seg in batch:
            seg["tag"] = results.get(seg["id"], "UNDECIDED")
