# Zyntral self-hosted voice service (OpenVoice v2 + MeloTTS, CPU).
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg pkg-config build-essential curl unzip ca-certificates \
        libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
        libavfilter-dev libswscale-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch (smaller, no CUDA)
RUN pip install --no-cache-dir torch==2.2.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# OpenVoice + MeloTTS (installed from source; not cleanly on PyPI)
RUN git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git /opt/OpenVoice \
    && pip install --no-cache-dir -e /opt/OpenVoice
RUN git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git /opt/MeloTTS \
    && pip install --no-cache-dir -e /opt/MeloTTS \
    && python -m unidic download
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')" || true

# OpenVoice v2 checkpoints
RUN curl -L -o /tmp/ckpt.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip \
    && unzip -q /tmp/ckpt.zip -d /app && rm /tmp/ckpt.zip

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

COPY app.py /app/app.py

ENV OPENVOICE_CKPT=/app/checkpoints_v2 VOICE_DATA_DIR=/data/voices
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
