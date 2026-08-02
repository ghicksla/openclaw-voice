#!/usr/bin/env python3
"""
Download Whisper and semantic turn models for offline use.

Usage:
    python scripts/download_models.py [model_name]
    python scripts/download_models.py smart-turn

Models:
    tiny    - 39M params, ~1GB VRAM (fastest)
    base    - 74M params, ~1GB VRAM
    small   - 244M params, ~2GB VRAM
    medium  - 769M params, ~5GB VRAM
    large-v3-turbo - 809M params, ~6GB VRAM (best quality/speed)
"""

import sys
import urllib.request
from pathlib import Path

SMART_TURN_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/" "smart-turn-v3.2-cpu.onnx"
)
SMART_TURN_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "smart-turn"


def download_smart_turn() -> bool:
    """Download the pinned Smart Turn v3.2 CPU checkpoint."""
    destination = SMART_TURN_MODEL_DIR / SMART_TURN_URL.rsplit("/", 1)[-1]
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"Smart Turn model already present: {destination}")
        return True
    try:
        SMART_TURN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Smart Turn model to {destination}")
        urllib.request.urlretrieve(SMART_TURN_URL, destination)
        if destination.stat().st_size < 1_000_000:
            raise OSError("Downloaded Smart Turn model is unexpectedly small")
        print("Smart Turn model downloaded successfully")
        return True
    except Exception as exc:
        destination.unlink(missing_ok=True)
        print(f"Failed to download Smart Turn model: {exc}")
        return False


def download_model(model_name: str = "base"):
    """Download a Whisper model."""
    print(f"Downloading Whisper model: {model_name}")
    print("This may take a few minutes...")

    try:
        from faster_whisper import WhisperModel

        # This will download the model if not cached
        model = WhisperModel(model_name, device="cpu", compute_type="int8")

        print(f"✅ Model '{model_name}' downloaded successfully!")
        print("   Cached at: ~/.cache/huggingface/")

        # Test the model
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)
        segments, info = model.transcribe(audio)
        list(segments)  # Consume generator

        print("✅ Model tested successfully!")

    except ImportError:
        print("❌ faster-whisper not installed. Run:")
        print("   pip install faster-whisper")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def list_models():
    """List available models."""
    models = {
        "tiny": "39M params, ~1GB VRAM, fastest",
        "base": "74M params, ~1GB VRAM, good balance",
        "small": "244M params, ~2GB VRAM",
        "medium": "769M params, ~5GB VRAM",
        "large-v3": "1.5B params, ~10GB VRAM, best quality",
        "large-v3-turbo": "809M params, ~6GB VRAM, best quality/speed ratio",
    }

    print("Available Whisper models:")
    print()
    for name, desc in models.items():
        print(f"  {name:20} - {desc}")
    print()
    print("Recommended: large-v3-turbo (GPU) or base (CPU)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        list_models()
        print()
        model = input("Enter model name to download (or 'q' to quit): ").strip()
        if model.lower() == "q":
            sys.exit(0)
    else:
        model = sys.argv[1]

        if model in ["-h", "--help", "help"]:
            list_models()
            sys.exit(0)

        if model in ("smart-turn", "smart_turn"):
            sys.exit(0 if download_smart_turn() else 1)

    download_model(model)
