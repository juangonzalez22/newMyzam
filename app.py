import os
import sys
import subprocess
from flask import Flask, render_template, request, jsonify

# Asegurar que Flask encuentre tus scripts en la carpeta 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_audio
from spectrogram import generate_spectrogram
from peaks import detect_peaks
from fingerprints import generate_fingerprints
from database import FingerprintDatabase
from matcher import recognize_song

app = Flask(__name__)

AMP_MIN = -35
NEIGHBORHOOD_SIZE = 30
FAN_VALUE = 10
DB_PATH = os.path.join(os.path.dirname(__file__), 'src', 'fingerprints.db')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'audio' not in request.files:
        return jsonify({'error': 'No se recibió audio'}), 400
    
    audio_file = request.files['audio']
    
    # Creamos rutas para el archivo temporal del navegador y el WAV final que lee Scipy
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_webm_path = os.path.join(base_dir, 'temp_query.webm')
    final_wav_path = os.path.join(base_dir, 'query.wav')
    
    # 1. Guardamos el blob crudo del navegador tal cual viene
    audio_file.save(temp_webm_path)
    
    # Conectar a la base de datos de huellas
    db = FingerprintDatabase(DB_PATH)
    
    try:
        # 2. CONVERSIÓN BLINDADA: Forzamos a FFmpeg a reparar las marcas de tiempo de Android
        cmd_convert = [
            "static_ffmpeg", "-y",
            "-fflags", "+genpts",       # <-- REPARA EL DESORDEN CRONOLÓGICO DE ANDROID
            "-i", temp_webm_path,
            "-vn",                      # Ignora metadatos o pistas basura
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            final_wav_path
        ]
        subprocess.run(cmd_convert, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 3. Pipeline matemático intacto leyendo el WAV reparado
        y, sr = preprocess_audio(final_wav_path)
        if y is None:
            return jsonify({'error': 'Error al procesar el audio'}), 400
        
        spec = generate_spectrogram(y)
        peaks = detect_peaks(spec, amp_min=AMP_MIN, neighborhood_size=NEIGHBORHOOD_SIZE)
        fingerprints = generate_fingerprints(peaks, fan_value=FAN_VALUE)
        
        if len(fingerprints) == 0:
            return jsonify({'match': False, 'message': 'Audio demasiado silencioso o ruidoso.'})
        
        result, score = recognize_song(fingerprints, db)
        
        if result is None:
            return jsonify({'match': False, 'message': 'No se encontraron coincidencias.'})
        
        predicted_song, offset = result
        return jsonify({
            'match': True,
            'song': predicted_song,
            'score': int(score),
            'offset': round(float(offset), 2)
        })
        
    except Exception as e:
        return jsonify({'error': f"Error en procesamiento: {str(e)}"}), 500
    finally:
        db.close()
        # Limpieza higiénica de los archivos temporales en el disco de tu PC
        if os.path.exists(temp_webm_path): os.remove(temp_webm_path)
        if os.path.exists(final_wav_path): os.remove(final_wav_path)

if __name__ == '__main__':
    # Configurado en 0.0.0.0 para que escuche a tu teléfono por Wi-Fi
    app.run(host='0.0.0.0', port=5000, debug=True)