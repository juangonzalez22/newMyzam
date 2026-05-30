import os
import time
import sys

# Intentamos importar la API nativa de Android dentro de Pydroid 3
try:
    import androidhelper
    droid = androidhelper.Android()
except ImportError:
    # Fallback por si estás probando el script en PC antes de pasarlo al móvil
    droid = None

def record_audio_android(filename="query.wav", duration=10):
    """
    Graba audio directamente desde el micrófono de Android en tiempo real
    usando la API interna del sistema a través de Pydroid 3.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_path = os.path.join(base_dir, filename)

    # Eliminar query anterior si existe
    if os.path.exists(audio_path):
        os.remove(audio_path)

    if droid is None:
        print("\n[MÁQUINA LOCAL] No estás en Android. Creando un archivo simulado o usa el método anterior.")
        return filename

    print("\n" + "=" * 70)
    print("RECONOCIMIENTO EN TIEMPO REAL")
    print("=" * 70)
    print("Presiona [ENTER] para empezar a escuchar el ambiente...")
    input()

    print("🎙️  ESCUCHANDO... (Graba durante 10 segundos)")
    
    try:
        # Iniciamos la grabación nativa de Android. 
        # Nota: Guarda directamente en el contenedor del sistema.
        droid.recorderStartMicrophone(audio_path)
        
        # Cuenta regresiva visual en la consola
        for remaining in range(duration, 0, -1):
            sys.stdout.write(f"\rFaltan {remaining} segundos... ")
            sys.stdout.flush()
            time.sleep(1)
            
        # Detener la grabación de golpe
        droid.recorderStop()
        print("\n\n✅ Grabación finalizada con éxito.")
        
    except Exception as e:
        print(f"\n[ERROR CRÍTICO AL GRABAR]: {e}")
        print("Asegúrate de que Pydroid 3 tenga permisos de MICRÓFONO otorgados en los ajustes de tu celular.")
        
    return filename