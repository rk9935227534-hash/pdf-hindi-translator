import os
import uuid
import shutil
import tempfile
import threading
from pathlib import Path

import fitz
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).parent
WORK_DIR = Path(os.getenv("WORK_DIR", APP_DIR / "work"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

MAX_PAGES = int(os.getenv("MAX_PAGES", "150"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "30"))
LIBRETRANSLATE_URL = os.getenv(
    "LIBRETRANSLATE_URL", "https://libretranslate.com/translate"
)
LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", "")
FONT_PATH = os.getenv(
    "HINDI_FONT_PATH",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"
)

app = FastAPI(title="PDF Hindi Translator")
jobs = {}
lock = threading.Lock()


def translate_text(text: str) -> str:
    """Translate one text block. Configure a reliable LibreTranslate endpoint."""
    if not text.strip():
        return text
    payload = {
        "q": text,
        "source": "en",
        "target": "hi",
        "format": "text",
    }
    if LIBRETRANSLATE_API_KEY:
        payload["api_key"] = LIBRETRANSLATE_API_KEY

    response = requests.post(LIBRETRANSLATE_URL, data=payload, timeout=90)
    response.raise_for_status()
    translated = response.json().get("translatedText")
    if not translated:
        raise RuntimeError("Translation service returned an empty result.")
    return translated


def fit_text(page, rect, text, fontname="hindi", fontsize=9):
    """Insert translated text inside the original block area, shrinking if needed."""
    box = fitz.Rect(rect)
    # A small inset avoids touching neighbouring lines or borders.
    box.x0 += 1
    box.x1 -= 1
    box.y0 += 0.5
    box.y1 -= 0.5

    size = max(5.5, min(fontsize, 12))
    while size >= 5.5:
        remaining = page.insert_textbox(
            box, text, fontname=fontname, fontsize=size,
            color=(0, 0, 0), lineheight=1.12, overlay=True
        )
        if remaining >= 0:
            return True
        # Remove any partial insertion before retrying.
        # insert_textbox does not insert if it cannot fit the complete text.
        size -= 0.5
    return False


def process_pdf(input_path: Path, output_path: Path, job_id: str):
    try:
        if not Path(FONT_PATH).exists():
            raise RuntimeError(
                "Hindi font not found. Install Noto Sans Devanagari and set "
                "HINDI_FONT_PATH to its .ttf file."
            )

        doc = fitz.open(input_path)
        total_pages = len(doc)
        with lock:
            jobs[job_id]["total_pages"] = total_pages

        for page_index, page in enumerate(doc):
            # Get text blocks in reading order. Images and vector artwork remain.
            blocks = page.get_text("dict")["blocks"]
            replacements = []
            for block in blocks:
                if block.get("type") != 0:
                    continue
                lines = block.get("lines", [])
                spans = [span for line in lines for span in line.get("spans", [])]
                original = "".join(span.get("text", "") for span in spans).strip()
                if not original:
                    continue

                bbox = block.get("bbox")
                max_size = max((s.get("size", 9) for s in spans), default=9)
                try:
                    hindi = translate_text(original)
                except Exception as exc:
                    raise RuntimeError(
                        f"Translation failed on page {page_index + 1}: {exc}"
                    ) from exc
                replacements.append((bbox, hindi, max_size))

            # Redact all original text areas, then draw Hindi text back.
            # Keep images and vector graphics. Nearby complex layouts may need review.
            for bbox, _, _ in replacements:
                page.add_redact_annot(fitz.Rect(bbox), fill=(1, 1, 1))
            if replacements:
                page.apply_redactions(images=0, graphics=0, text=0)

            failed_blocks = 0
            for bbox, hindi, size in replacements:
                if not fit_text(page, bbox, hindi, fontsize=size):
                    failed_blocks += 1

            with lock:
                jobs[job_id]["completed_pages"] = page_index + 1
                jobs[job_id]["message"] = (
                    f"पेज {page_index + 1}/{total_pages} प्रोसेस हो चुका है"
                )
                jobs[job_id]["overflow_blocks"] = (
                    jobs[job_id].get("overflow_blocks", 0) + failed_blocks
                )

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        with lock:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "अनुवाद पूरा हुआ।"
    except Exception as exc:
        with lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["message"] = str(exc)


@app.get("/")
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.post("/api/translate")
async def start_translation(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "कृपया PDF फाइल अपलोड करें।")

    job_id = uuid.uuid4().hex
    input_path = WORK_DIR / f"{job_id}_input.pdf"
    output_path = WORK_DIR / f"{job_id}_hindi.pdf"

    size = 0
    with input_path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_MB * 1024 * 1024:
                input_path.unlink(missing_ok=True)
                raise HTTPException(413, f"फाइल {MAX_FILE_MB} MB से छोटी होनी चाहिए।")
            out.write(chunk)

    try:
        doc = fitz.open(input_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, "यह PDF खुल नहीं रही है या खराब है।")

    if page_count < 1 or page_count > MAX_PAGES:
        input_path.unlink(missing_ok=True)
        raise HTTPException(400, f"PDF में 1 से {MAX_PAGES} पेज होने चाहिए।")

    with lock:
        jobs[job_id] = {
            "status": "processing",
            "completed_pages": 0,
            "total_pages": page_count,
            "message": "अनुवाद शुरू हो रहा है...",
            "overflow_blocks": 0,
        }

    background_tasks.add_task(process_pdf, input_path, output_path, job_id)
    return {"job_id": job_id, "total_pages": page_count}


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "यह काम नहीं मिला।")
        return job


@app.get("/api/download/{job_id}")
def download(job_id: str):
    with lock:
        job = jobs.get(job_id)
        if not job or job["status"] != "completed":
            raise HTTPException(409, "PDF अभी तैयार नहीं है।")
    path = WORK_DIR / f"{job_id}_hindi.pdf"
    if not path.exists():
        raise HTTPException(404, "तैयार PDF उपलब्ध नहीं है।")
    return FileResponse(path, media_type="application/pdf",
                        filename="Hindi_Translated.pdf")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
