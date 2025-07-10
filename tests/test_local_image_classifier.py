import argparse
from pathlib import Path
import logging
import sys
import os

try:
    # Import from src now that project root is in path
    from src.local_image_classifier import classify_text_type_local, MODEL_NAME, MIN_CONFIDENCE_SCORE
except ImportError as e:
    print(f"Errore: Impossibile importare src.local_image_classifier. Assicurati che esista src/local_image_classifier.py e che le dipendenze siano installate.")
    print(f"Dettagli errore: {e}")
    # sys.exit(1) # Avoid sys.exit in a test module that might be imported by unittest
    raise # Reraise the import error so unittest can catch it if this file is treated as a test

# Configure basic logging to see output from the classifier module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# You can set a lower level (e.g., logging.DEBUG) to see more detailed logs 
# from the classifier, like the full recognized text, if needed.
# logging.getLogger("local_image_classifier").setLevel(logging.DEBUG) 

logger = logging.getLogger(__name__)

# TODO: Convert this script to a unittest.TestCase class if it needs to be run by 'unittest discover'

def main():
    parser = argparse.ArgumentParser(description="Testa la classificazione locale di immagini usando DiT (RVL-CDIP).")
    parser.add_argument("image_file", type=Path, help="Percorso del file immagine da classificare.")
    
    args = parser.parse_args()

    if not args.image_file.is_file():
        logger.error(f"File immagine non trovato: {args.image_file}")
        sys.exit(1)

    logger.info(f"--- Inizio Test Classificazione Locale con DiT ---")
    logger.info(f"Modello Utilizzato: {MODEL_NAME}")
    logger.info(f"Soglia di confidenza minima per classificazione primaria: {MIN_CONFIDENCE_SCORE}")
    logger.info(f"File Immagine: {args.image_file.name}")

    primary_class, all_predictions = classify_text_type_local(args.image_file)

    logger.info(f"--- Risultato Classificazione --- endeavoured")
    if primary_class:
        logger.info(f"File: {args.image_file.name}")
        logger.info(f"  -> Classificazione Primaria: '{primary_class}'")
        if all_predictions:
            logger.info(f"  -> Tutte le Predizioni DiT (top 5):")
            for pred in all_predictions:
                logger.info(f"     - Label: {pred['label']:<25} Score: {pred['score']:.4f}")
        else:
            logger.info("  -> Nessuna predizione dettagliata disponibile da DiT.")
    else:
        logger.error(f"Classificazione fallita per {args.image_file.name}. Controllare i log precedenti per errori.")
    
    logger.info(f"--- Fine Test Classificazione Locale --- endeavoured")

if __name__ == "__main__":
    main() 