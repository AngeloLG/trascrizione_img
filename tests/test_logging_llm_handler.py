import logging
from pathlib import Path
import sys
import os

# 1. Configurazione del Logging (identica a quella che dovrebbe funzionare)
# PROJECT_ROOT_FOR_LOGGING = Path(__file__).resolve().parent # This was relative to the old script location
# Use the correctly calculated project_root_calculated for consistency if needed, or define logs relative to execution
LOG_FILE_NAME_FOR_LOGGING = "test_llm_logging.log" 
# Make log directory in project root for this specific test script

# project_root needs to be defined for LOG_DIRECTORY_FOR_LOGGING if used before imports
# For a script in tests/ directory, project_root is its parent.
project_root = Path(__file__).resolve().parent.parent
LOG_DIRECTORY_FOR_LOGGING = project_root / "logs_test_llm_standalone"

LOG_DIRECTORY_FOR_LOGGING.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIRECTORY_FOR_LOGGING / LOG_FILE_NAME_FOR_LOGGING, mode='w', encoding='utf-8'), # mode 'w' per sovrascrivere
        logging.StreamHandler(sys.stdout)
    ]
)

# Assicurati che gli handler abbiano il livello corretto
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
    elif isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.DEBUG)

# Ottieni il logger per questo script di test
test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.DEBUG) # Assicurati che anche questo logger sia a DEBUG

test_logger.info("Logging configurato per test_logging_llm_handler.py")
test_logger.debug("Questo è un messaggio DEBUG dallo script di test.")

# 2. Importa SOLO le parti necessarie da llm_handler
# Assicurati che sys.path sia corretto se src non è direttamente importabile
# Se esegui dalla root del progetto, `from src.llm_handler...` dovrebbe funzionare
try:
    # Assumendo che llm_handler.py sia in una directory 'src'
    # e che lo script di test sia nella root del progetto
    # e che src/__init__.py esista
    from src.llm_handler import get_transcription_from_local_ocr, _initialize_local_ocr_model
    from src import llm_handler # Per accedere al suo logger
    
    # Forza il logger di llm_handler a DEBUG (già fatto dentro llm_handler.py, ma ridondante per sicurezza)
    llm_handler_logger = logging.getLogger(llm_handler.__name__)
    llm_handler_logger.setLevel(logging.DEBUG)
    test_logger.debug(f"Livello del logger {llm_handler.__name__} impostato a: {llm_handler_logger.getEffectiveLevel()} (DEBUG={logging.DEBUG})")

    test_logger.info("Tentativo di inizializzare il modello OCR locale...")
    if _initialize_local_ocr_model():
        test_logger.info("Modello OCR locale inizializzato con successo.")
        
        # 3. Simula una chiamata alla funzione problematica
        #    Crea un finto Path per un'immagine esistente per il test
        #    MODIFICA QUESTO PERCORSO CON UN'IMMAGINE JPG/JPEG REALE CHE USI PER I TEST
        fake_image_path = project_root / "input_immagini/artom008/image_103.jpg"
        
        if fake_image_path.exists() and fake_image_path.suffix.lower() in ['.jpg', '.jpeg']:
            test_logger.info(f"Tentativo di chiamare get_transcription_from_local_ocr con: {fake_image_path}")
            transcription = get_transcription_from_local_ocr(fake_image_path)
            if transcription is not None:
                test_logger.info(f"Trascrizione ricevuta (potrebbe essere '0 0' o altro): '{transcription}'")
            else:
                test_logger.error("get_transcription_from_local_ocr ha restituito None.")
        else:
            test_logger.error(f"L'immagine di test specificata non esiste o non è JPG/JPEG: {fake_image_path}")
            test_logger.error("PER FAVORE, MODIFICA 'fake_image_path' NELLO SCRIPT test_logging_llm_handler.py CON UN PERCORSO VALIDO SE QUESTO FALLISCE.")

    else:
        test_logger.error("Fallita inizializzazione del modello OCR locale.")

except ImportError as e:
    test_logger.error(f"Errore di importazione: {e}. Assicurati che 'src' sia nel Python path.")
except Exception as e:
    test_logger.error(f"Errore imprevisto durante l'esecuzione dello script di test: {e}", exc_info=True)

test_logger.info("Esecuzione di test_logging_llm_handler.py completata.") 