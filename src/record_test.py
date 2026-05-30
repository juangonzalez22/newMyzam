import os
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np

from preprocess import preprocess_audio
from spectrogram import generate_spectrogram
from peaks import detect_peaks
from fingerprints import generate_fingerprints


from database import FingerprintDatabase
from matcher import recognize_song


# =========================
# CONFIG
# =========================

SAMPLE_RATE = 44100
CHANNELS = 1

AMP_MIN = -35
NEIGHBORHOOD_SIZE = 30
FAN_VALUE = 10


# =========================
# RECORD AUDIO
# =========================

def record_audio():

    print("\n" + "=" * 70)
    print("PRESS ENTER TO START RECORDING")
    input()

    print("RECORDING...")
    print("PRESS ENTER TO STOP")

    recording = []

    def callback(indata, frames, time, status):

        if status:
            print(status)

        recording.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback
    )

    stream.start()

    input()

    stream.stop()
    stream.close()

    print("RECORDING FINISHED")

    audio = np.concatenate(recording, axis=0)

    # =========================
    # SAVE TEMP FILE
    # =========================

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    )

    temp_path = temp_file.name

    temp_file.close()

    sf.write(
        temp_path,
        audio,
        SAMPLE_RATE
    )

    return temp_path


# =========================
# PROCESS QUERY
# =========================

def process_query(path):
    """
    Procesa query completa.
    """

    y, sr = preprocess_audio(path)

    spec = generate_spectrogram(y)

    peaks = detect_peaks(
        spec,
        amp_min=AMP_MIN,
        neighborhood_size=NEIGHBORHOOD_SIZE
    )

    fingerprints = generate_fingerprints(
        peaks,
        fan_value=FAN_VALUE
    )

    return fingerprints


# =========================
# MAIN
# =========================

def main():

    db = FingerprintDatabase()

    print("=" * 70)
    print("SONG RECOGNITION SYSTEM")
    print("=" * 70)

    try:

        while True:

            # =========================
            # RECORD
            # =========================

            temp_path = record_audio()

            # =========================
            # PROCESS
            # =========================

            fingerprints = process_query(temp_path)

            # =========================
            # MATCH
            # =========================

            result, score = recognize_song(
                fingerprints,
                db
            )

            print("\n" + "-" * 70)

            if result is None:

                print("RESULT: NO MATCH")

            else:

                predicted_song, offset = result

                print(f"PREDICTED: {predicted_song}")
                print(f"SCORE: {score}")
                print(f"OFFSET: {offset}")

            print("-" * 70)

            # =========================
            # DELETE TEMP FILE
            # =========================

            if os.path.exists(temp_path):
                os.remove(temp_path)

            print("\nPRESS:")
            print("[ENTER] -> NEW QUERY")
            print("[CTRL + C] -> EXIT")

            input()

    except KeyboardInterrupt:

        print("\nEXITING...")

    finally:

        db.close()


if __name__ == "__main__":
    main()
    