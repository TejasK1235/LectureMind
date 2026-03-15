# api.py
# LectureMind — FastAPI server exposing all NLP/AI pipelines
#
# Endpoints:
#   POST /transcribe       video/audio file → raw transcript JSON
#   POST /tag              raw transcript JSON → tagged JSON
#   POST /summarize        tagged JSON → markdown summary string
#   POST /question-bank    tagged JSON + optional slide file → QB JSON

import json
import os   
import tempfile 
import shutil
from contextlib import asynccontextmanager
from typing import Optional, List
import markdown
import pdfkit
from fastapi.responses import StreamingResponse
import io

import whisper
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Pipeline imports ──────────────────────────────────────────────────────────
from tagging2.pipeline.run import run_tagging_pipeline
from summarization.run import run_summarization_pipeline
from question_bank.run import run_qb_pipeline
from quiz.run import run_mcq_quiz, run_extempore_quiz


# ── Whisper segment merging ───────────────────────────────────────────────────
# Identical to your transcription script — kept here so the API is self-contained
MIN_SEGMENT_WORDS = 5
MAX_SEGMENT_WORDS = 15
SENTENCE_ENDINGS = (".", "?", "!", "...", "।")


def _merge_short_segments(segments: list) -> list:
    merged = []
    buffer_text = ""
    buffer_start = None

    def flush(end_time):
        merged.append({
            "start": buffer_start,
            "end": end_time,
            "text": buffer_text
        })

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if buffer_start is None:
            buffer_start = seg["start"]
        buffer_text = (buffer_text + " " + text).strip()
        word_count = len(buffer_text.split())
        ends_sentence = buffer_text.rstrip().endswith(SENTENCE_ENDINGS)
        has_enough_words = word_count >= MIN_SEGMENT_WORDS
        hit_hard_cap = word_count >= MAX_SEGMENT_WORDS

        if (ends_sentence and has_enough_words) or hit_hard_cap:
            flush(seg["end"])
            buffer_text = ""
            buffer_start = None

    if buffer_text:
        if merged:
            last = merged[-1]
            merged[-1] = {
                "start": last["start"],
                "end": segments[-1]["end"],
                "text": last["text"] + " " + buffer_text
            }
        else:
            merged.append({
                "start": buffer_start,
                "end": segments[-1]["end"],
                "text": buffer_text
            })

    return merged


# ── Load Whisper once at startup, reuse across all requests ───────────────────
WHISPER_MODEL_SIZE = "small"
whisper_model = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global whisper_model
#     print(f"[Startup] Loading Whisper ({WHISPER_MODEL_SIZE})...")
#     whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
#     print("[Startup] Ready.")
#     yield
#     print("[Shutdown] Server stopping.")
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Startup] Loading Whisper ({WHISPER_MODEL_SIZE}) on {device}...")

    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)

    print("Model device:", next(whisper_model.parameters()).device)
    print("[Startup] Ready.")

    yield

    print("[Shutdown] Server stopping.")


app = FastAPI(
    title="LectureMind NLP API",
    version="1.0.0",
    lifespan=lifespan
)

# @app.on_event("startup")
# def load_model():
#     global whisper_model

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Loading Whisper on {device}")

#     whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)


# ── Helper: write UploadFile to a temp file, return path ─────────────────────
def _save_upload(upload: UploadFile, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(upload.file, tmp)
    finally:
        tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: /transcribe
#
# Input  (multipart/form-data):
#   lecture_file  — audio or video file (required)
#
# Output (JSON):
#   {
#     "segment_count": 220,
#     "segments": [
#       {"start": 0.0, "end": 6.06, "text": "..."},
#       ...
#     ]
#   }
#
# ─────────────────────────────────────────────────────────────────────────────
# @app.post("/transcribe")
# async def transcribe(
#     lecture_file: UploadFile = File(...),
#     # slide_file: Optional[UploadFile] = File(None)
# ):
#     ext = os.path.splitext(lecture_file.filename or "lecture")[1].lower() or ".wav"
#     tmp_path = None
#     try:
#         tmp_path = _save_upload(lecture_file, suffix=ext)
#         # result = whisper_model.transcribe(tmp_path, task="translate", verbose=False)
#         result = whisper_model.transcribe(tmp_path, task="translate", verbose=False, fp16=torch.cuda.is_available())
#         raw_segments = result.get("segments", [])
#         merged = _merge_short_segments(raw_segments)

#         # Exact same format as your transcripts.json files
#         segments = [
#             {
#                 "start": round(seg["start"], 2),
#                 "end":   round(seg["end"], 2),
#                 "text":  seg["text"]
#             }
#             for seg in merged
#         ]

#         print(f"[/transcribe] {len(raw_segments)} raw → {len(segments)} merged segments")
#         return JSONResponse(content={
#             "segment_count": len(segments),
#             "segments": segments
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             os.remove(tmp_path)

@app.post("/transcribe")
async def transcribe(
    lecture_file: UploadFile = File(...),
):
    ext = os.path.splitext(lecture_file.filename or "lecture")[1].lower() or ".wav"
    tmp_path = None
    try:
        tmp_path = _save_upload(lecture_file, suffix=ext)
        result = whisper_model.transcribe(tmp_path, task="translate", verbose=False, fp16=torch.cuda.is_available())
        raw_segments = result.get("segments", [])
        merged = _merge_short_segments(raw_segments)

        segments = [
            {
                "start": round(seg["start"], 2),
                "end":   round(seg["end"], 2),
                "text":  seg["text"]
            }
            for seg in merged
        ]

        print(f"[/transcribe] {len(raw_segments)} raw - {len(segments)} merged segments")

        # Chain tagging directly — backend expects tagged output from this endpoint
        print(f"[/transcribe] Running tagging pipeline on merged segments...")
        tagged = run_tagging_pipeline(segments)

        from collections import Counter
        dist = Counter(s.get("tag") for s in tagged)
        print(f"[/transcribe] Tagging done: {dict(dist)}")

        return JSONResponse(content=tagged)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: /tag
#
# Input (JSON body):
#   {
#     "segments": [
#       {"start": 0.0, "end": 6.06, "text": "..."},
#       ...
#     ]
#   }
#
# Output (JSON):
#   {
#     "segment_count": 339,
#     "segments": [
#       {
#         "id": "seg_29d9f304",
#         "start_time": 0.0,
#         "end_time": 6.06,
#         "text": "...",
#         "tag": "LECTURE_CONTENT",
#         "emphasis": false
#       },
#       ...
#     ]
#   }
#
# Note: segment.py inside the tagger converts start→start_time, end→end_time
# and assigns an "id". This tagged format is what /summarize and /question-bank expect.
# ─────────────────────────────────────────────────────────────────────────────
class TranscriptRequest(BaseModel):
    segments: List[dict]


@app.post("/tag")
async def tag(request: TranscriptRequest):
    if not request.segments:
        raise HTTPException(status_code=400, detail="No segments provided.")
    try:
        tagged = run_tagging_pipeline(request.segments)

        from collections import Counter
        dist = Counter(s.get("tag") for s in tagged)
        print(f"[/tag] {len(tagged)} segments tagged: {dict(dist)}")

        return JSONResponse(content=tagged)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tagging failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: /summarize
#
# Input (JSON body):
#   {
#     "segments": [
#       {"id": "seg_xxx", "start_time": 0.0, "end_time": 6.06,
#        "text": "...", "tag": "LECTURE_CONTENT", "emphasis": false},
#       ...
#     ]
#   }
#   — exactly the "segments" array from /tag output
#
# Output (JSON):
#   {
#     "lecture_segment_count": 339,
#     "summary": "## File Systems\n\n- Point one\n- Point two\n..."
#   }
#
# ─────────────────────────────────────────────────────────────────────────────
class TaggedRequest(BaseModel):
    segments: List[dict]


# @app.post("/summarize")
# async def summarize(request: TaggedRequest):
#     if not request.segments:
#         raise HTTPException(status_code=400, detail="No segments provided.")

#     lecture_segments = [s for s in request.segments if s.get("tag") == "LECTURE_CONTENT"]
#     if not lecture_segments:
#         raise HTTPException(status_code=400, detail="No LECTURE_CONTENT segments found.")

#     try:
#         summary = run_summarization_pipeline(request.segments)
#         print(f"[/summarize] {len(summary)} chars from {len(lecture_segments)} lecture segments.")
#         return JSONResponse(content={
#             "lecture_segment_count": len(lecture_segments),
#             "summary": summary
#         })
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/summarize")
async def summarize(request: TaggedRequest):
    if not request.segments:
        raise HTTPException(status_code=400, detail="No segments provided.")

    lecture_segments = [s for s in request.segments if s.get("tag") == "LECTURE_CONTENT"]
    if not lecture_segments:
        raise HTTPException(status_code=400, detail="No LECTURE_CONTENT segments found.")

    try:
        summary_md = run_summarization_pipeline(request.segments)
        print(f"[/summarize] {len(summary_md)} chars from {len(lecture_segments)} lecture segments.")
        print(f"[/summarize] Summary content:\n{summary_md}\n")

        # Convert markdown → HTML → PDF
        html_body = markdown.markdown(summary_md, extensions=["extra", "tables"])

        html_full = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 13px;
                line-height: 1.7;
                margin: 48px 56px;
                color: #1a1a1a;
            }}
            h1 {{ font-size: 22px; color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; margin-top: 32px; }}
            h2 {{ font-size: 17px; color: #1a3a5c; margin-top: 24px; }}
            h3 {{ font-size: 14px; color: #333; margin-top: 16px; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 5px; }}
            code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
            pre {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }}
            strong {{ color: #1a3a5c; }}
        </style>
        </head>
        <body>
        {html_body}
        </body>
        </html>
        """

        # config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
        # pdf_bytes = pdfkit.from_string(html_full, False, configuration=config)
        config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
        options = {
            "page-size": "A4",
            "encoding": "UTF-8",
            "no-outline": None,
            "enable-local-file-access": None,
        }
        pdf_bytes = pdfkit.from_string(html_full, False, configuration=config, options=options)
        pdf_buffer = io.BytesIO(pdf_bytes)

        print(f"[/summarize] PDF generated ({len(pdf_bytes)} bytes).")

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=lecture_summary.pdf"}
        )

    except Exception as e:
        print(f"[/summarize] ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    
# @app.post("/summarize")
# async def summarize(request: Request):
#     body = await request.body()
#     print(f"[/summarize DEBUG] Raw body: {body[:500]}")  # first 500 chars so it doesnt flood terminal
#     print(f"[/summarize DEBUG] Content-Type: {request.headers.get('content-type')}")
#     raise HTTPException(status_code=200, detail="debug")



@app.post("/summarize/with-slides")
async def summarize_with_slides(
    segments_json: str = Form(...),
    slide_file: UploadFile = File(...),
):
    try:
        segments = json.loads(segments_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid segments_json: {e}")

    if not segments:
        raise HTTPException(status_code=400, detail="No segments provided.")

    lecture_segments = [s for s in segments if s.get("tag") == "LECTURE_CONTENT"]
    if not lecture_segments:
        raise HTTPException(status_code=400, detail="No LECTURE_CONTENT segments found.")

    ext = os.path.splitext(slide_file.filename or "")[1].lower()
    if ext not in (".pdf", ".pptx", ".ppt"):
        raise HTTPException(status_code=400, detail=f"Unsupported slide format: {ext}")

    slide_tmp_path = None
    try:
        slide_tmp_path = _save_upload(slide_file, suffix=ext)
        print(f"[/summarize/with-slides] Slide file received: {slide_file.filename}")

        summary_md = run_summarization_pipeline(segments, slide_file_path=slide_tmp_path)
        print(f"[/summarize/with-slides] {len(summary_md)} chars from {len(lecture_segments)} lecture segments.")

        html_body = markdown.markdown(summary_md, extensions=["extra", "tables"])
        html_full = f"""
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                font-size: 13px;
                line-height: 1.7;
                margin: 48px 56px;
                color: #1a1a1a;
            }}
            h1 {{ font-size: 22px; color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 6px; margin-top: 32px; }}
            h2 {{ font-size: 17px; color: #1a3a5c; margin-top: 24px; }}
            h3 {{ font-size: 14px; color: #333; margin-top: 16px; }}
            ul {{ padding-left: 20px; }}
            li {{ margin-bottom: 5px; }}
            code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
            pre {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }}
            strong {{ color: #1a3a5c; }}
        </style>
        </head>
        <body>
        {html_body}
        </body>
        </html>
        """
        config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
        pdf_bytes = pdfkit.from_string(html_full, False, configuration=config)
        pdf_buffer = io.BytesIO(pdf_bytes)
        print(f"[/summarize/with-slides] PDF generated ({len(pdf_bytes)} bytes).")

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=lecture_summary.pdf"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
    finally:
        if slide_tmp_path and os.path.exists(slide_tmp_path):
            os.remove(slide_tmp_path)

# ── Shared helper so both QB endpoints don't repeat the same logic ────────────
class QBRequest(BaseModel):
    segments: List[dict]
    total_questions: int = 20
    bloom_remember:   float = 0.2
    bloom_understand: float = 0.2
    bloom_apply:      float = 0.2
    bloom_analyze:    float = 0.2
    bloom_evaluate:   float = 0.2


def _build_bloom(request: QBRequest) -> dict:
    bloom_percentages = {
        "remember":   request.bloom_remember,
        "understand": request.bloom_understand,
        "apply":      request.bloom_apply,
        "analyze":    request.bloom_analyze,
        "evaluate":   request.bloom_evaluate,
    }
    bloom_percentages = {k: v for k, v in bloom_percentages.items() if v > 0}
    total_pct = sum(bloom_percentages.values())
    if not (0.98 <= total_pct <= 1.02):
        raise HTTPException(
            status_code=400,
            detail=f"Bloom percentages must sum to 1.0, got {total_pct:.2f}"
        )
    return bloom_percentages


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4a: /question-bank
#
# Input (JSON body):
#   {
#     "segments": [...],        ← tagged segments array from /tag
#     "total_questions": 20,    ← optional, default 20
#     "bloom_remember":   0.2,  ← optional, all default to 0.2
#     "bloom_understand": 0.2,
#     "bloom_apply":      0.2,
#     "bloom_analyze":    0.2,
#     "bloom_evaluate":   0.2
#   }
#
# Output: same QB JSON as before
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/question-bank")
async def question_bank(request: QBRequest):
    if not request.segments:
        raise HTTPException(status_code=400, detail="No segments provided.")

    bloom_percentages = _build_bloom(request)

    try:
        result = run_qb_pipeline(
            tagged_segments=request.segments,
            total_questions=request.total_questions,
            bloom_percentages=bloom_percentages,
            slide_file_path=None
        )
        print(f"[/question-bank] {result['total_questions']} questions. Warnings: {result['warnings']}")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QB generation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4b: /question-bank/with-slides
#
# Input (multipart/form-data):
#   segments_json     — tagged segments array, JSON stringified (required)
#   slide_file        — PDF or PPTX file (required, that's the whole point)
#   total_questions   — integer, default 20 (optional)
#   bloom_*           — floats summing to 1.0, default 0.2 each (optional)
#
# Output: same QB JSON as /question-bank
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/question-bank/with-slides")
async def question_bank_with_slides(
    segments_json:    str          = Form(...),
    slide_file:       UploadFile   = File(...),
    total_questions:  int          = Form(20),
    bloom_remember:   float        = Form(0.2),
    bloom_understand: float        = Form(0.2),
    bloom_apply:      float        = Form(0.2),
    bloom_analyze:    float        = Form(0.2),
    bloom_evaluate:   float        = Form(0.2),
):
    try:
        segments = json.loads(segments_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid segments_json: {str(e)}")

    if not segments:
        raise HTTPException(status_code=400, detail="No segments provided.")

    # Reuse the same bloom validation via a throwaway QBRequest
    dummy = QBRequest(
        segments=segments,
        total_questions=total_questions,
        bloom_remember=bloom_remember,
        bloom_understand=bloom_understand,
        bloom_apply=bloom_apply,
        bloom_analyze=bloom_analyze,
        bloom_evaluate=bloom_evaluate,
    )
    bloom_percentages = _build_bloom(dummy)

    ext = os.path.splitext(slide_file.filename or "")[1].lower()
    if ext not in (".pdf", ".pptx", ".ppt"):
        raise HTTPException(status_code=400, detail=f"Unsupported slide format: {ext}. Use PDF or PPTX.")

    slide_tmp_path = None
    try:
        slide_tmp_path = _save_upload(slide_file, suffix=ext)
        print(f"[/question-bank/with-slides] Slide file received: {slide_file.filename}")

        result = run_qb_pipeline(
            tagged_segments=segments,
            total_questions=total_questions,
            bloom_percentages=bloom_percentages,
            slide_file_path=slide_tmp_path
        )
        print(f"[/question-bank/with-slides] {result['total_questions']} questions. Warnings: {result['warnings']}")
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QB generation failed: {str(e)}")
    finally:
        if slide_tmp_path and os.path.exists(slide_tmp_path):
            os.remove(slide_tmp_path)



# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 5: /quiz/mcq
#
# Input (JSON body):
#   {
#     "qb_result": { ...full QB output JSON including "concepts" key... },
#     "num_questions": 10      ← optional, default 10
#   }
#
# Output:
#   {
#     "total": 8,
#     "mcqs": [{question, options, correct, correct_text, bloom_level}, ...],
#     "warnings": [...]
#   }
# ─────────────────────────────────────────────────────────────────────────────
class MCQRequest(BaseModel):
    qb_result: dict
    num_questions: int = 10


@app.post("/quiz/mcq")
async def quiz_mcq(request: MCQRequest):
    concepts = request.qb_result.get("concepts", [])
    if not concepts:
        raise HTTPException(
            status_code=400,
            detail="No concepts found in qb_result. Make sure QB pipeline was run with updated run_qb_pipeline()."
        )

    questions_exist = any(
        request.qb_result.get("questions", {}).get(level)
        for level in ["remember", "understand"]
    )
    if not questions_exist:
        raise HTTPException(
            status_code=400,
            detail="No Remember or Understand level questions found in qb_result."
        )

    try:
        result = run_mcq_quiz(
            qb_result=request.qb_result,
            concepts=concepts,
            num_questions=request.num_questions
        )
        print(f"[/quiz/mcq] {result['total']} MCQs generated. Warnings: {result['warnings']}")
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCQ generation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 6: /quiz/extempore
#
# Input (JSON body):
#   {
#     "qb_result": { ...full QB output JSON including "concepts" key... },
#     "num_topics": 5          ← optional, default 5
#   }
#
# Output:
#   {
#     "total": 5,
#     "topics": [{concept_id, title, score, word_count}, ...],
#     "warnings": [...]
#   }
# ─────────────────────────────────────────────────────────────────────────────
class ExtemporeRequest(BaseModel):
    qb_result: dict
    num_topics: int = 5


@app.post("/quiz/extempore")
async def quiz_extempore(request: ExtemporeRequest):
    concepts = request.qb_result.get("concepts", [])
    if not concepts:
        raise HTTPException(
            status_code=400,
            detail="No concepts found in qb_result. Make sure QB pipeline was run with updated run_qb_pipeline()."
        )

    try:
        result = run_extempore_quiz(concepts)

        # Slice to requested num_topics here at the API layer
        top_topics = result["topics"][:request.num_topics]

        response = {
            "total": len(top_topics),
            "topics": top_topics,
            "warnings": result["warnings"]
        }

        print(f"[/quiz/extempore] {len(top_topics)} topics returned. Warnings: {result['warnings']}")
        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extempore generation failed: {str(e)}")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "whisper_model": WHISPER_MODEL_SIZE}