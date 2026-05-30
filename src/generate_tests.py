"""
generate_tests.py
─────────────────
Genera fragmentos de prueba aleatorios a partir de la carpeta 'songs'.
Por cada canción crea N fragmentos de 5-10 s con filtros de audio variados.
Todos los resultados se guardan como WAV en la carpeta 'queries'.
"""

import os
import random
import time
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from scipy.ndimage import uniform_filter1d

# ──────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────
CARPETA_SONGS   = "songs"
CARPETA_OUTPUT  = "queries"
FRAGMENTOS_POR_CANCION = 4

DURACION_MIN_SEG = 5.0
DURACION_MAX_SEG = 10.0
SAMPLE_RATE      = 22050        # SR interno de trabajo (más calidad que el de analysis.py)
FORMATOS_ENTRADA = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

# Semilla para reproducibilidad (pon None para resultados distintos cada vez)
SEMILLA_ALEATORIA = None


# ──────────────────────────────────────────
# CATÁLOGO DE FILTROS
# ──────────────────────────────────────────

def filtro_ruido_blanco(y: np.ndarray, sr: int) -> np.ndarray:
    """Añade ruido blanco gaussiano de baja amplitud."""
    nivel = random.uniform(0.002, 0.025)
    ruido = np.random.normal(0, nivel, len(y))
    return np.clip(y + ruido, -1.0, 1.0)


def filtro_ruido_ambiente(y: np.ndarray, sr: int) -> np.ndarray:
    """Simula ruido de fondo tipo 'room noise' (ruido rosa aproximado)."""
    # Ruido rosa: filtrar ruido blanco con decaída 1/f
    ruido_blanco = np.random.normal(0, 1, len(y))
    # Aproximación de ruido rosa con filtro pasa-bajos acumulativo
    ruido_rosa   = uniform_filter1d(ruido_blanco, size=int(sr * 0.01))
    ruido_rosa  /= (np.max(np.abs(ruido_rosa)) + 1e-9)
    nivel        = random.uniform(0.01, 0.06)
    return np.clip(y + ruido_rosa * nivel, -1.0, 1.0)


def filtro_bajos_recortados(y: np.ndarray, sr: int) -> np.ndarray:
    """Simula grabación con micrófono barato: corta las bajas frecuencias."""
    frecuencia_corte = random.uniform(150, 400)    # Hz
    sos = butter(4, frecuencia_corte / (sr / 2), btype='high', output='sos')
    return sosfilt(sos, y).astype(np.float32)


def filtro_telefono(y: np.ndarray, sr: int) -> np.ndarray:
    """Simula calidad de audio de llamada telefónica (300–3400 Hz)."""
    sos_hp = butter(4, 300  / (sr / 2), btype='high', output='sos')
    sos_lp = butter(4, 3400 / (sr / 2), btype='low',  output='sos')
    y2 = sosfilt(sos_hp, y)
    y2 = sosfilt(sos_lp, y2)
    return y2.astype(np.float32)


def filtro_compresion_mp3(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Simula artefactos de compresión MP3 con cuantización de coeficientes DCT.
    Rápido y sin necesidad de codificadores externos.
    """
    from scipy.fft import dct, idct
    bloque   = int(sr * 0.023)        # bloques de ~23 ms (tamaño MP3 estándar)
    n_coefs  = random.randint(20, 60) # cuántos coeficientes conservar (menos = peor calidad)
    resultado = np.zeros_like(y)

    for inicio in range(0, len(y) - bloque, bloque):
        seg = y[inicio: inicio + bloque]
        C   = dct(seg, norm='ortho')
        C[n_coefs:] = 0               # descarta altas frecuencias (como hace MP3)
        resultado[inicio: inicio + bloque] = idct(C, norm='ortho')

    return np.clip(resultado, -1.0, 1.0).astype(np.float32)


def filtro_eco(y: np.ndarray, sr: int) -> np.ndarray:
    """Añade un eco simple con retardo y atenuación aleatorios."""
    retardo_ms  = random.uniform(80, 300)
    atenuacion  = random.uniform(0.15, 0.40)
    retardo_muestras = int(sr * retardo_ms / 1000)
    eco         = np.zeros_like(y)
    eco[retardo_muestras:] = y[:-retardo_muestras] * atenuacion
    return np.clip(y + eco, -1.0, 1.0)


def filtro_volumen(y: np.ndarray, sr: int) -> np.ndarray:
    """Sube o baja el volumen general de forma aleatoria."""
    factor = random.uniform(0.3, 1.4)
    return np.clip(y * factor, -1.0, 1.0)


def filtro_saturacion(y: np.ndarray, sr: int) -> np.ndarray:
    """Saturación/distorsión suave (soft-clip)."""
    ganancia = random.uniform(1.5, 4.0)
    y_g      = y * ganancia
    # Soft-clip con tanh
    return np.tanh(y_g * random.uniform(0.8, 1.5)).astype(np.float32)


def filtro_graves_reforzados(y: np.ndarray, sr: int) -> np.ndarray:
    """Refuerza bajas frecuencias (efecto 'bass boost')."""
    fc  = random.uniform(80, 200)
    sos = butter(2, fc / (sr / 2), btype='low', output='sos')
    bajos = sosfilt(sos, y).astype(np.float32)
    nivel = random.uniform(0.3, 0.8)
    return np.clip(y + bajos * nivel, -1.0, 1.0)


# ──────────────────────────────────────────
# REGISTRO DE FILTROS (nombre → función)
# ──────────────────────────────────────────

FILTROS_DISPONIBLES = {
    "ruido_blanco"       : filtro_ruido_blanco,
    "ruido_ambiente"     : filtro_ruido_ambiente,
    "bajos_recortados"   : filtro_bajos_recortados,
    "telefono"           : filtro_telefono,
    "compresion_mp3"     : filtro_compresion_mp3,
    "eco"                : filtro_eco,
    "volumen"            : filtro_volumen,
    "saturacion"         : filtro_saturacion,
    "graves_reforzados"  : filtro_graves_reforzados,
}


# ──────────────────────────────────────────
# PIPELINE DE GENERACIÓN
# ──────────────────────────────────────────

def aplicar_cadena_filtros(y: np.ndarray, sr: int, nombres_filtros: list) -> np.ndarray:
    """Aplica secuencialmente los filtros indicados."""
    for nombre in nombres_filtros:
        try:
            y = FILTROS_DISPONIBLES[nombre](y, sr)
        except Exception as e:
            print(f"      ⚠ Filtro '{nombre}' falló: {e}. Saltando.")
    return y


def elegir_filtros_aleatorios(n_min: int = 1, n_max: int = 3) -> list:
    """
    Selecciona entre n_min y n_max filtros al azar.
    Evita combinar filtros incompatibles (ej: teléfono + graves reforzados).
    """
    n = random.randint(n_min, n_max)
    disponibles = list(FILTROS_DISPONIBLES.keys())

    # Siempre incluye ruido o volumen para asegurar algo de variación perceptible
    obligatorio = random.choice(["ruido_blanco", "ruido_ambiente", "volumen"])
    seleccion   = [obligatorio]

    resto = [f for f in disponibles if f != obligatorio]
    random.shuffle(resto)
    seleccion += resto[:n - 1]

    # Evitar combinación telefono + graves (se anulan mutuamente)
    if "telefono" in seleccion and "graves_reforzados" in seleccion:
        seleccion.remove("graves_reforzados")

    return seleccion


def generar_fragmento(ruta_cancion: str, indice: int) -> dict | None:
    """
    Carga la canción, recorta un fragmento aleatorio y aplica filtros.
    Devuelve un dict con metadatos o None si hay error.
    """
    nombre_base = os.path.splitext(os.path.basename(ruta_cancion))[0]

    try:
        y, sr = librosa.load(ruta_cancion, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"   ✗ No se pudo cargar: {e}")
        return None

    duracion_total = len(y) / sr

    # La canción debe ser lo suficientemente larga
    if duracion_total < DURACION_MIN_SEG + 1:
        print(f"   ⚠ La canción es demasiado corta ({duracion_total:.1f}s). Saltando.")
        return None

    # ── Elegir fragmento aleatorio ──
    duracion_frag = random.uniform(DURACION_MIN_SEG, DURACION_MAX_SEG)
    duracion_frag = min(duracion_frag, duracion_total - 0.5)  # no pasar del final
    max_inicio    = duracion_total - duracion_frag
    inicio_seg    = random.uniform(0, max_inicio)

    inicio_muestra = int(inicio_seg * sr)
    fin_muestra    = inicio_muestra + int(duracion_frag * sr)
    fragmento      = y[inicio_muestra:fin_muestra].copy()

    # ── Elegir y aplicar filtros ──
    filtros_elegidos = elegir_filtros_aleatorios(n_min=1, n_max=3)
    fragmento_filtrado = aplicar_cadena_filtros(fragmento, sr, filtros_elegidos)

    # Normalizar para evitar clipping al guardar
    pico = np.max(np.abs(fragmento_filtrado))
    if pico > 1e-6:
        fragmento_filtrado = fragmento_filtrado / pico * 0.92

    # ── Nombre del archivo de salida ──
    etiqueta_filtros = "+".join(f[:4] for f in filtros_elegidos)   # abreviatura legible
    nombre_salida    = f"{nombre_base}__frag{indice:02d}__{etiqueta_filtros}.wav"

    return {
        "audio"         : fragmento_filtrado,
        "sr"            : sr,
        "nombre_salida" : nombre_salida,
        "inicio_seg"    : inicio_seg,
        "duracion_seg"  : duracion_frag,
        "filtros"       : filtros_elegidos,
    }


# ──────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────

def main():
    if SEMILLA_ALEATORIA is not None:
        random.seed(SEMILLA_ALEATORIA)
        np.random.seed(SEMILLA_ALEATORIA)

    # Preparar carpetas
    os.makedirs(CARPETA_OUTPUT, exist_ok=True)

    canciones = sorted(
        f for f in os.listdir(CARPETA_SONGS)
        if os.path.splitext(f)[1].lower() in FORMATOS_ENTRADA
    )

    if not canciones:
        print(f"No se encontraron canciones en '{CARPETA_SONGS}'.")
        return

    total_esperado = len(canciones) * FRAGMENTOS_POR_CANCION
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  GENERADOR DE PRUEBAS DE AUDIO                       ║")
    print(f"║  Canciones encontradas : {len(canciones):<28}║")
    print(f"║  Fragmentos por canción: {FRAGMENTOS_POR_CANCION:<28}║")
    print(f"║  Total esperado        : {total_esperado:<28}║")
    print(f"║  Carpeta de salida     : {CARPETA_OUTPUT:<28}║")
    print(f"╚══════════════════════════════════════════════════════╝\n")

    t0             = time.perf_counter()
    total_creados  = 0
    total_fallidos = 0
    log_lineas     = []   # para el resumen final

    for num_cancion, nombre_cancion in enumerate(canciones, start=1):
        ruta = os.path.join(CARPETA_SONGS, nombre_cancion)
        print(f"[{num_cancion}/{len(canciones)}] {nombre_cancion}")

        for i in range(1, FRAGMENTOS_POR_CANCION + 1):
            resultado = generar_fragmento(ruta, i)

            if resultado is None:
                total_fallidos += 1
                continue

            ruta_salida = os.path.join(CARPETA_OUTPUT, resultado["nombre_salida"])
            sf.write(ruta_salida, resultado["audio"], resultado["sr"], subtype='PCM_16')

            filtros_str = " → ".join(resultado["filtros"])
            inicio      = resultado["inicio_seg"]
            dur         = resultado["duracion_seg"]

            print(f"   [{i}/{FRAGMENTOS_POR_CANCION}] {resultado['nombre_salida']}")
            print(f"          Inicio: {inicio:.1f}s  Duración: {dur:.1f}s")
            print(f"          Filtros: {filtros_str}")

            log_lineas.append(
                f"{resultado['nombre_salida']} | origen: {nombre_cancion} | "
                f"inicio: {inicio:.1f}s | dur: {dur:.1f}s | filtros: {filtros_str}"
            )
            total_creados += 1

        print()

    # Guardar log en texto plano
    ruta_log = os.path.join(CARPETA_OUTPUT, "_resumen_pruebas.txt")
    with open(ruta_log, "w", encoding="utf-8") as f:
        f.write(f"RESUMEN DE PRUEBAS GENERADAS — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
        for linea in log_lineas:
            f.write(linea + "\n")

    duracion_total = time.perf_counter() - t0
    print("═" * 56)
    print(f"  ✓ Fragmentos creados : {total_creados}")
    print(f"  ✗ Fallidos           : {total_fallidos}")
    print(f"  ⏱ Tiempo total       : {duracion_total:.1f}s")
    print(f"  📁 Carpeta           : {CARPETA_OUTPUT}/")
    print(f"  📄 Log               : {ruta_log}")
    print("═" * 56)


if __name__ == "__main__":
    main()