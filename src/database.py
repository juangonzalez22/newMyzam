import sqlite3


class FingerprintDatabase:

    def __init__(self, db_path="fingerprints.db"):
        """
        Inicializa conexión SQLite.
        """

        self.conn = sqlite3.connect(db_path)

        self.cursor = self.conn.cursor()

        self.create_tables()
        
    def create_tables(self):
        """
        Crea tablas e índices.
        """

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            hash TEXT,
            song_id TEXT,
            offset INTEGER
        )
        """)

        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hash
        ON fingerprints(hash)
        """)

        self.conn.commit()
        
    def add_song(self,
                 song_id,
                 fingerprints):
        """
        Inserta fingerprints de una canción.
        """

        rows = [
            (h, song_id, int(offset))
            for h, offset in fingerprints
        ]

        self.cursor.executemany("""
        INSERT INTO fingerprints
        (hash, song_id, offset)
        VALUES (?, ?, ?)
        """, rows)

        self.conn.commit()
        
    def query_hash(self, h):
        """
        Busca coincidencias para un hash.
        """

        self.cursor.execute("""
        SELECT song_id, offset
        FROM fingerprints
        WHERE hash = ?
        """, (h,))

        return self.cursor.fetchall()
    
    def count_fingerprints(self):
        """
        Cuenta fingerprints almacenados.
        """

        self.cursor.execute("""
        SELECT COUNT(*)
        FROM fingerprints
        """)

        return self.cursor.fetchone()[0]

    def close(self):

        self.conn.close()
        
    