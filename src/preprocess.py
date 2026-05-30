import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

TARGET_SR = 22050

def load_audio(path, target_sr=TARGET_SR):
    """Carga archivo de audio de forma nativa sin librosa."""
    sr, y = wav.read(path)
    
    # Normalizar a float32 entre -1.0 y 1.0
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    elif y.dtype == np.uint8:
        y = (y.astype(np.float32) - 128.0) / 128.0
        
    # Convertir a mono
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # Remuestreo si es necesario
    if sr != target_sr:
        up = 1
        down = sr // target_sr
        if down > 0 and sr % target_sr == 0:
            y = signal.resample_poly(y, up, down)
        else:
            # Fallback simple si la proporción no es exacta
            num_samples = int(len(y) * target_sr / sr)
            y = signal.resample(y, num_samples)
        sr = target_sr

    return y.astype(np.float32), sr

def normalize_audio(y):
    """Normaliza amplitud entre -1 y 1."""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y

def preprocess_audio(path):
    """Preprocesa audio de forma robusta."""
    try:
        y, sr = load_audio(path)
        if len(y) == 0:
            raise ValueError("Audio vacío")
        y = normalize_audio(y)
        return y, sr
    except Exception as e:
        print(f"[ERROR] {path}: {e}")
        return None, None