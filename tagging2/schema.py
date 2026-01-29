# schema.py
from typing import TypedDict, Optional


class TranscriptSegment(TypedDict):
    id: str
    start_time: float
    end_time: float
    text: str


class TaggedSegment(TranscriptSegment):
    tag: str
