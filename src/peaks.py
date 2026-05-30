import numpy as np
from scipy.ndimage import maximum_filter

def detect_peaks(spectrogram, amp_min=-30, neighborhood_size=35):
    """
    Detecta picos espectrales robustos usando Scipy nativo.
    """
    # Máximos locales (Tu lógica exacta, impecable)
    local_max = maximum_filter(
        spectrogram,
        size=neighborhood_size
    ) == spectrogram

    # Threshold amplitud
    detected_peaks = local_max & (spectrogram > amp_min)

    # Coordenadas
    peaks = np.argwhere(detected_peaks)

    return peaks


def plot_peaks(spectrogram, peaks, sr, hop_length):
    """
    Visualiza peaks sobre espectrograma.
    NOTA: Esta función solo se ejecutará si estás en PC con Librosa instalado.
    En Android se ignorará de forma segura para evitar crashes.
    """
    try:
        import matplotlib.pyplot as plt
        import librosa.display
        
        plt.figure(figsize=(14, 6))

        librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='log'
        )

        plt.scatter(
            peaks[:, 1],
            peaks[:, 0],
            color='red',
            s=10
        )

        plt.title("Detected Peaks")
        plt.show()
    except ImportError:
        print("[AVISO] Entorno sin interfaz gráfica (Android). Saltando visualización de picos.")