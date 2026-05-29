"""
Audio preprocessing for improved STT accuracy.

Pipeline (applied before Whisper):
1. High-pass filter — removes sub-80Hz rumble (HVAC, desk vibration, handling noise)
2. Spectral noise reduction — noisereduce library, adaptive spectral gating
3. Normalization — peak-normalize to consistent level for Whisper
4. Silence trimming — remove leading/trailing silence

All operations work on float32 numpy arrays at 16kHz.
"""

import numpy as np
from scipy.signal import butter, sosfilt
from loguru import logger

# Pre-compute high-pass filter coefficients (80Hz cutoff at 16kHz sample rate)
# Using 4th-order Butterworth — gentle rolloff, no ringing artifacts
_HPF_SOS = butter(4, 80, btype='high', fs=16000, output='sos')

# Lazy-load noisereduce (heavy import, pulls matplotlib)
_nr = None


def _get_noisereduce():
    global _nr
    if _nr is None:
        try:
            import noisereduce as nr
            _nr = nr
            logger.info("✅ noisereduce loaded for audio preprocessing")
        except ImportError:
            logger.warning("noisereduce not installed — skipping spectral noise reduction")
            _nr = False  # sentinel: don't retry
    return _nr if _nr is not False else None


def high_pass_filter(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Remove frequencies below 80Hz (rumble, handling noise)."""
    if len(audio) < 100:
        return audio
    return sosfilt(_HPF_SOS, audio).astype(np.float32)


def reduce_noise(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Spectral noise reduction via noisereduce.

    Uses stationary noise reduction — estimates noise profile from the
    quietest parts of the signal and subtracts it spectrally.
    prop_decrease=0.8 keeps some natural room tone to avoid robotic artifacts.
    """
    nr = _get_noisereduce()
    if nr is None:
        return audio
    try:
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            prop_decrease=0.8,       # 80% noise reduction (not 100% — avoids artifacts)
            stationary=True,          # assume stationary noise (fan, AC, etc.)
            n_fft=512,                # smaller FFT for 16kHz (faster, still effective)
            hop_length=128,
        )
        return reduced.astype(np.float32)
    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}")
        return audio


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalize audio to a consistent level for Whisper."""
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio  # silence, don't amplify noise floor
    return (audio * (target_peak / peak)).astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int = 16000, threshold_db: float = -40.0) -> np.ndarray:
    """
    Trim leading and trailing silence.

    Uses energy-based detection with a -40dB threshold.
    Keeps 100ms padding on each end for natural speech edges.
    """
    if len(audio) < sr // 10:  # less than 100ms, don't trim
        return audio

    # Convert threshold from dB to linear
    threshold = 10 ** (threshold_db / 20.0)

    # Compute frame energy (10ms frames)
    frame_len = sr // 100  # 160 samples at 16kHz = 10ms
    n_frames = len(audio) // frame_len
    if n_frames < 3:
        return audio

    energies = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    # Find first and last frames above threshold
    active = np.where(energies > threshold)[0]
    if len(active) == 0:
        return audio  # all silence — return as-is, let Whisper handle it

    first = max(0, active[0] - 10)  # 100ms padding (10 frames)
    last = min(n_frames - 1, active[-1] + 10)

    start_sample = first * frame_len
    end_sample = min(len(audio), (last + 1) * frame_len)

    return audio[start_sample:end_sample]


def preprocess(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Full preprocessing pipeline for STT.

    Order matters:
    1. High-pass filter (removes rumble before noise estimation)
    2. Noise reduction (spectral gating on clean-ish signal)
    3. Trim silence (remove dead air)
    4. Normalize (consistent level for Whisper)
    """
    if len(audio) < sr // 4:  # less than 250ms, skip processing
        return audio

    original_len = len(audio)
    audio = high_pass_filter(audio, sr)
    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio, sr)
    audio = normalize(audio, target_peak=0.9)

    trimmed = original_len - len(audio)
    if trimmed > sr:  # log if we trimmed more than 1 second
        logger.debug(f"Audio preprocessed: {original_len/sr:.1f}s → {len(audio)/sr:.1f}s")

    return audio
