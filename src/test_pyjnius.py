"""
test_pyjnius.py
---------------
Prueba mínima para verificar que pyjnius puede acceder
a la API de audio de Android.

Ejecutar en Android (via terminal o antes de compilar el APK).
"""

from jnius import autoclass

# =========================
# CONSTANTES DE ANDROID
# =========================

AudioRecord    = autoclass("android.media.AudioRecord")
AudioFormat    = autoclass("android.media.AudioFormat")
AudioSource    = autoclass("android.media.MediaRecorder$AudioSource")
AudioManager   = autoclass("android.media.AudioManager")

SAMPLE_RATE    = 44100
CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
AUDIO_FORMAT   = AudioFormat.ENCODING_PCM_16BIT

# =========================
# TEST 1: buffer mínimo
# =========================

min_buffer = AudioRecord.getMinBufferSize(
    SAMPLE_RATE,
    CHANNEL_CONFIG,
    AUDIO_FORMAT
)

print(f"[TEST 1] Min buffer size: {min_buffer}")

if min_buffer <= 0:
    print("[FAIL] getMinBufferSize devolvió valor inválido")
else:
    print("[OK] getMinBufferSize funcionó")

# =========================
# TEST 2: crear instancia
# =========================

try:
    recorder = AudioRecord(
        AudioSource.MIC,
        SAMPLE_RATE,
        CHANNEL_CONFIG,
        AUDIO_FORMAT,
        min_buffer
    )

    state = recorder.getState()
    print(f"[TEST 2] Estado del recorder: {state}")

    # AudioRecord.STATE_INITIALIZED == 1
    if state == 1:
        print("[OK] AudioRecord inicializado correctamente")
    else:
        print("[FAIL] AudioRecord no se inicializó (¿permiso de micrófono?)")

    recorder.release()

except Exception as e:
    print(f"[FAIL] No se pudo crear AudioRecord: {e}")