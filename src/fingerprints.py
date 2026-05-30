import hashlib


def hash_peak_pair(f1, f2, delta_t):
    """
    Genera hash para un par de peaks.
    """

    data = f"{f1}|{f2}|{delta_t}"

    return hashlib.sha1(data.encode()).hexdigest()[:20]


def generate_fingerprints(peaks,
                           fan_value=10):
    """
    Genera fingerprints estilo Shazam.
    """

    fingerprints = []

    # peaks viene como:
    # [freq_bin, time_bin]

    for i in range(len(peaks)):

        anchor_freq = peaks[i][0]
        anchor_time = peaks[i][1]

        # conectar con siguientes peaks
        for j in range(1, fan_value):

            if i + j < len(peaks):

                target_freq = peaks[i + j][0]
                target_time = peaks[i + j][1]

                delta_t = target_time - anchor_time

                # evitar negativos o cero
                if delta_t <= 0:
                    continue

                h = hash_peak_pair(
                    anchor_freq,
                    target_freq,
                    delta_t
                )

                fingerprints.append(
                    (h, anchor_time)
                )

    return fingerprints