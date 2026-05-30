from collections import defaultdict


def get_matches(fingerprints,
                db):
    """
    Obtiene matches desde SQLite.
    """

    matches = []

    for h, query_offset in fingerprints:

        db_matches = db.query_hash(h)

        for song_id, song_offset in db_matches:

            matches.append(
                (
                    song_id,
                    song_offset,
                    query_offset
                )
            )

    return matches

def align_matches(matches):
    """
    Construye histogramas de offsets.
    """

    offset_counts = defaultdict(int)

    for song_id, song_offset, query_offset in matches:

        delta = song_offset - query_offset

        offset_counts[(song_id, delta)] += 1

    return offset_counts


def score_matches(offset_counts):
    """
    Encuentra mejor match.
    """

    best_match = None
    best_score = 0

    for key, count in offset_counts.items():

        if count > best_score:

            best_score = count
            best_match = key

    return best_match, best_score


def recognize_song(fingerprints,
                   db):
    """
    Reconocimiento completo.
    """

    matches = get_matches(
        fingerprints,
        db
    )

    offset_counts = align_matches(
        matches
    )

    best_match, score = score_matches(
        offset_counts
    )

    return best_match, score