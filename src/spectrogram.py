import numpy as np
import scipy.signal as signal

N_FFT = 2048
HOP_LENGTH = 512
TARGET_SR = 22050

def compute_stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Calcula STFT compleja usando Scipy con compatibilidad de bordes."""
    nperseg = n_fft
    noverlap = n_fft - hop_length
    
    # Cambiado 'reflection' por 'even' para compatibilidad estricta con tu versión de Scipy
    f, t, Zxx = signal.stft(
        y, 
        fs=TARGET_SR, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap, 
        boundary='even'
    )
    return Zxx

def magnitude_spectrogram(stft):
    """Convierte STFT compleja en magnitud."""
    return np.abs(stft)

def db_spectrogram(spectrogram):
    """Convierte amplitud a escala logarítmica (dB)."""
    spectrogram = np.maximum(spectrogram, 1e-10)
    db_spec = 20 * np.log10(spectrogram)
    return db_spec - np.max(db_spec)
    
def generate_spectrogram(y):
    """Pipeline completo espectrograma."""
    stft = compute_stft(y)
    magnitude = magnitude_spectrogram(stft)
    db_spec = db_spectrogram(magnitude)
    return db_spec