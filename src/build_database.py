import os

from preprocess import preprocess_audio
from spectrogram import generate_spectrogram
from peaks import detect_peaks
from fingerprints import generate_fingerprints

from database import FingerprintDatabase


SONGS_DIR = "songs"


db = FingerprintDatabase()


files = os.listdir(SONGS_DIR)


for file in files:

    path = os.path.join(SONGS_DIR, file)

    print("=" * 50)
    print(f"Procesando: {file}")

    y, sr = preprocess_audio(path)

    if y is None:
        continue

    spec = generate_spectrogram(y)

    peaks = detect_peaks(
        spec,
        amp_min=-35,
        neighborhood_size=30
    )

    print(f"Peaks: {len(peaks)}")

    fingerprints = generate_fingerprints(
        peaks,
        fan_value=10
    )

    print(f"Fingerprints: {len(fingerprints)}")

    db.add_song(file, fingerprints)

    print("Guardado en SQLite")


print("=" * 50)

print(
    "Total fingerprints:",
    db.count_fingerprints()
)

db.close()