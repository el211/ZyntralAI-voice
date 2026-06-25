# Zyntral Voice Service (self-hosted CPU voice cloning)

OpenVoice v2 + MeloTTS wrapped in a small FastAPI service. MIT-licensed models,
no per-use fees. Runs on CPU (slow: ~15–60s per generation). For speed, run on a GPU host.

## Endpoints
- `GET  /health` → `{"status":"ok"}`
- `POST /clone` (multipart `file`: an audio sample, ~30–120s clean speech) → `{"voiceId":"<id>"}`
- `POST /tts` (JSON `{ "text": "...", "voiceId": "<id>", "language": "EN" }`) → `audio/wav`

## Deploy in Coolify
1. Put this folder in a GitHub repo (e.g. `el211/ZyntralAI-voice`).
2. Coolify → + New → Application → from your repo, Build Pack = Dockerfile,
   Base Directory `/voice-service` (or repo root if that's where the Dockerfile is).
3. **Persistent Storage:** mount a volume at `/data` so cloned voices survive redeploys.
4. **Resource limits:** give it as much RAM as you can (≥4GB recommended).
5. **Healthcheck path:** `/health`, port `8000`.
6. (Optional) bind an internal domain; the Zyntral backend calls it via
   `VOICE_SERVICE_URL` (e.g. `http://<service>:8000`).

First boot is slow (downloads model checkpoints during the image build).

## Notes
- CPU inference is slow and memory-hungry. If it OOMs or is too slow, this needs a GPU host.
- Cloned voice embeddings are tiny `.pth` files under `/data/voices`.
