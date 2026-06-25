"""
Zyntral self-hosted voice service — CPU voice cloning + TTS via OpenVoice v2 (MIT).

Endpoints:
  GET  /health                      -> {"status":"ok"}
  POST /clone   (multipart: file)   -> {"voiceId": "..."}   extract & store a target voice
  POST /tts     (json: text,voiceId,language) -> audio/wav  synthesize text in that voice

Cloned voice embeddings are stored on disk under DATA_DIR/<voiceId>.pth so they persist
(mount a volume there). CPU inference is slow; keep texts short.
"""
import os
import uuid
import tempfile

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

DEVICE = "cpu"
CKPT = os.environ.get("OPENVOICE_CKPT", "/app/checkpoints_v2")
DATA_DIR = os.environ.get("VOICE_DATA_DIR", "/data/voices")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Zyntral Voice Service")

# Loaded once at startup (heavy).
converter = ToneColorConverter(f"{CKPT}/converter/config.json", device=DEVICE)
converter.load_ckpt(f"{CKPT}/converter/checkpoint.pth")

_tts_cache: dict[str, TTS] = {}


def tts_for(language: str) -> TTS:
    lang = (language or "EN").upper()
    if lang not in _tts_cache:
        _tts_cache[lang] = TTS(language=lang, device=DEVICE)
    return _tts_cache[lang]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/clone")
async def clone(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "sample.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        sample_path = tmp.name
    try:
        target_se, _ = se_extractor.get_se(sample_path, converter, vad=True)
        voice_id = uuid.uuid4().hex
        torch.save(target_se, os.path.join(DATA_DIR, f"{voice_id}.pth"))
        return {"voiceId": voice_id}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"clone failed: {e}")
    finally:
        try:
            os.remove(sample_path)
        except OSError:
            pass


class TtsRequest(BaseModel):
    text: str
    voiceId: str
    language: str | None = "EN"
    speed: float | None = 1.0


@app.post("/tts")
def tts(req: TtsRequest):
    se_path = os.path.join(DATA_DIR, f"{req.voiceId}.pth")
    if not os.path.exists(se_path):
        raise HTTPException(status_code=404, detail="voiceId not found")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        target_se = torch.load(se_path, map_location=DEVICE)
        model = tts_for(req.language or "EN")
        speaker_ids = model.hps.data.spk2id
        base_speaker = list(speaker_ids.values())[0]

        base_wav = os.path.join(tempfile.gettempdir(), f"base_{uuid.uuid4().hex}.wav")
        out_wav = os.path.join(tempfile.gettempdir(), f"out_{uuid.uuid4().hex}.wav")
        model.tts_to_file(req.text, base_speaker, base_wav, speed=req.speed or 1.0)

        source_se = torch.load(f"{CKPT}/base_speakers/ses/en-default.pth", map_location=DEVICE)
        converter.convert(audio_src_path=base_wav, src_se=source_se, tgt_se=target_se,
                          output_path=out_wav, message="@Zyntral")
        try:
            os.remove(base_wav)
        except OSError:
            pass
        return FileResponse(out_wav, media_type="audio/wav", filename="speech.wav")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"detail": f"tts failed: {e}"})
