import math
from scorer import compute_scores


def select_top_segments(segments, ratio=0.25):
    scores = compute_scores(segments)

    scored = list(zip(segments, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    k = math.ceil(len(scored) * ratio)
    selected = scored[:k]

    # Preserve original order
    selected_segments = [s for s, _ in selected]
    selected_segments.sort(key=lambda s: s["start_time"])

    return selected_segments
