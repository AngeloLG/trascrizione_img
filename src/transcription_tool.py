import logging
import sys # Per sys.stdout nel logging
from pathlib import Path # Per LOG_DIRECTORY nel logging

# --- Logger Setup PRIMA DI TUTTO ---
# Definisci le costanti necessarie per il logging qui, se non sono già globali e accessibili
PROJECT_ROOT_FOR_LOGGING = Path(__file__).resolve().parent.parent
LOG_FILE_NAME_FOR_LOGGING = "transcription_tool.log"
LOG_DIRECTORY_FOR_LOGGING = PROJECT_ROOT_FOR_LOGGING / "logs"

# Ensure log directory exists
LOG_DIRECTORY_FOR_LOGGING.mkdir(parents=True, exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIRECTORY_FOR_LOGGING / LOG_FILE_NAME_FOR_LOGGING, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set higher level for console output and ensure FileHandler is DEBUG
# È importante che logging.getLogger().handlers sia chiamato DOPO basicConfig
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.INFO)
    elif isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.DEBUG)

# Ora le altre importazioni
import argparse
import os
from typing import Optional, Tuple, List, Dict

from .file_handler import validate_image_file, save_transcription, load_prompt_file, SUPPORTED_IMAGE_EXTENSIONS
from .llm_handler import get_transcription_from_llm, OPENAI_API_CONFIGURED, get_transcription_from_local_ocr
from .local_image_classifier import classify_text_type_local

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent # Get the root of the project (up two levels from this file)
DEFAULT_PROMPT_FILENAME = "default_transcription_prompt.txt"
DEFAULT_PROMPT_FILE_PATH = PROJECT_ROOT / DEFAULT_PROMPT_FILENAME

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """
    Creates and configures the argument parser for the CLI.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Tool di Trascrizione: Trascrive testo da immagini JPG.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_file", type=Path, help="Percorso del singolo file immagine JPG da processare.")
    group.add_argument("--image_dir", type=Path, help="Percorso della directory contenente i file immagine JPG da processare.")
    
    parser.add_argument("--prompt_file", type=Path, default=None, 
                        help=f"Percorso del file contenente il prompt per la trascrizione. Default: ./{DEFAULT_PROMPT_FILENAME}")
    parser.add_argument("--output_dir", type=Path, default=None, 
                        help="Directory dove salvare i file di trascrizione. Default: stessa directory dell'immagine/immagini o directory di input per la modalità batch.")

    return parser

def get_prompt_content(prompt_file_path_arg: Optional[Path]) -> Optional[str]:
    """
    Loads the prompt content from a specified file or the default prompt file.

    Args:
        prompt_file_path_arg (Path | None): Path to a custom prompt file provided
                                            via CLI argument. If None, the default
                                            prompt is used.

    Returns:
        str | None: The content of the prompt file, or None if loading fails.
    """
    prompt_to_load: Path
    if prompt_file_path_arg:
        logger.info(f"Utilizzo del file prompt specificato: {prompt_file_path_arg}")
        prompt_to_load = prompt_file_path_arg
    else:
        logger.info(f"Nessun file prompt specificato, utilizzo del prompt di default: {DEFAULT_PROMPT_FILE_PATH}")
        prompt_to_load = DEFAULT_PROMPT_FILE_PATH

    prompt_content = load_prompt_file(prompt_to_load)
    if not prompt_content:
        logger.error(f"Impossibile caricare il contenuto del prompt da: {prompt_to_load}")
        if not prompt_to_load.exists() and prompt_to_load == DEFAULT_PROMPT_FILE_PATH:
            logger.error(f"Il file prompt di default ({DEFAULT_PROMPT_FILENAME}) non è stato trovato nella root del progetto.")
            logger.error(f"Assicurati che '{DEFAULT_PROMPT_FILENAME}' esista in: {PROJECT_ROOT}")
        return None
    return prompt_content

def process_single_image(
    image_path: Path,
    prompt_text: str,
    output_dir: Optional[Path]
) -> bool:
    """
    Processes a single image: validates, classifies, gets transcription, and saves it.

    Args:
        image_path (Path): Path to the image file.
        prompt_text (str): The prompt text to use for transcription.
        output_dir (Path | None): Directory to save the transcription.
                                  If None, defaults to the image's directory.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    logger.info(f"Inizio elaborazione per il file immagine singolo: {image_path}")

    if not validate_image_file(image_path):
        logger.error(f"Validazione fallita per {image_path}. Salto del file.")
        return False

    # --- Local Image Classification Step ---
    logger.info(f"Avvio classificazione locale tipo testo per: {image_path.name}")
    primary_classification, all_dit_predictions = classify_text_type_local(image_path)

    effective_image_type: str
    if primary_classification is None:
        logger.warning(f"Classificazione locale del tipo di immagine fallita per {image_path.name}. Sarà trattata come 'undetermined'.")
        effective_image_type = "undetermined"
    else:
        effective_image_type = primary_classification
        logger.info(f"Immagine {image_path.name} classificata localmente come (primaria): '{effective_image_type}'")
        if all_dit_predictions:
            logger.debug(f"  -> Tutte le Predizioni DiT per {image_path.name}: {all_dit_predictions}")
    
    transcription: Optional[str] = None

    if effective_image_type == "handwritten":
        logger.info(f"Rilevato testo manoscritto per {image_path.name}. Tentativo di trascrizione OCR locale con TrOCR.")
        transcription = get_transcription_from_local_ocr(image_path)
        if transcription is None:
            logger.error(f"Fallimento della trascrizione OCR locale per {image_path.name}. Valutare fallback o loggare errore.")
            # Potremmo decidere di fare un fallback a OpenAI qui, o semplicemente fallire per i manoscritti se l'OCR locale fallisce.
            # Per ora, se l'OCR locale fallisce, la trascrizione sarà None e il processo fallirà per questo file.
            return False 
        logger.info(f"Trascrizione OCR locale per {image_path.name} completata.")
    else:
        logger.info(f"Tipo immagine '{effective_image_type}' per {image_path.name}. Utilizzo API OpenAI per trascrizione.")
        if not OPENAI_API_CONFIGURED:
            logger.error("Configurazione API OpenAI fallita o non eseguita. Impossibile procedere con la trascrizione API.")
            return False
        # TODO: Qui si potrebbe usare all_dit_predictions per modificare il prompt_text se necessario
        # Esempio: 
        # modified_prompt_text = prompt_text
        # if effective_image_type == "form" and any(pred['label'] == 'handwritten' for pred in all_dit_predictions if pred['score'] > 0.3):
        #     modified_prompt_text = "Il seguente è un modulo compilato a mano. " + prompt_text 
        transcription = get_transcription_from_llm(image_path, prompt_text) 
        if transcription is None:
            logger.error(f"Fallimento nel ricevere la trascrizione da API OpenAI per {image_path.name}.")
            return False
        logger.info(f"Trascrizione da API OpenAI per {image_path.name} completata.")

    # A questo punto, 'transcription' dovrebbe contenere il testo o essere None se un passaggio precedente è fallito (e fatto return False)
    # Tuttavia, la logica sopra dovrebbe già fare return False in caso di fallimento della trascrizione.
    # Quindi, se arriviamo qui, la trascrizione dovrebbe essere una stringa (potenzialmente vuota dall'OCR).
    if transcription is None: # Doppia sicurezza, anche se la logica sopra dovrebbe prevenirlo
        logger.error(f"Trascrizione è None per {image_path.name} nonostante i controlli precedenti. Fallimento imprevisto.")
        return False

    logger.info(f"Trascrizione ottenuta per {image_path.name}. Tentativo di salvataggio.")
    saved_path = save_transcription(transcription, image_path, output_dir)
    if saved_path:
        logger.info(f"Trascrizione per {image_path.name} salvata con successo in {saved_path}")
        return True
    else:
        logger.error(f"Fallimento nel salvataggio della trascrizione per {image_path.name}.")
        return False

def process_directory(
    image_dir_path: Path,
    prompt_text: str,
    output_dir_arg: Optional[Path]
) -> bool:
    """
    Processes all supported image files in a given directory.

    Args:
        image_dir_path (Path): Path to the directory containing image files.
        prompt_text (str): The prompt text to use for transcription.
        output_dir_arg (Path | None): Directory to save transcriptions.
                                     If None, transcriptions are saved in the
                                     same directory as their respective images
                                     (if image_dir_path is used as output) or
                                     in a subdirectory within image_dir_path
                                     if output_dir_arg is not specified,
                                     defaulting to image_dir_path itself.

    Returns:
        bool: True if all images were processed successfully or no valid images were found.
              False if any image processing failed.
    """
    logger.info(f"Inizio elaborazione per la directory: {image_dir_path}")
    if not image_dir_path.is_dir():
        logger.error(f"Il percorso specificato per la directory non è una directory valida: {image_dir_path}")
        return False

    processed_files_count = 0
    successful_files_count = 0
    failed_files_count = 0
    found_image_files = False

    actual_output_dir = output_dir_arg if output_dir_arg else image_dir_path
    if output_dir_arg:
        logger.info(f"Directory di output specificata: {actual_output_dir}")
    else:
        logger.info(f"Nessuna directory di output specificata, utilizzo la directory di input: {actual_output_dir}")


    for item_path in image_dir_path.iterdir():
        if item_path.is_file() and item_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            found_image_files = True
            logger.info(f"Trovato file immagine supportato: {item_path.name}")
            processed_files_count += 1
            if process_single_image(item_path, prompt_text, actual_output_dir):
                successful_files_count += 1
            else:
                failed_files_count += 1
                logger.warning(f"Elaborazione fallita per {item_path.name} nella directory.")
        elif item_path.is_file():
            logger.debug(f"Saltato file non supportato o non immagine nella directory: {item_path.name}")

    if not found_image_files:
        logger.info(f"Nessun file immagine supportato ({', '.join(SUPPORTED_IMAGE_EXTENSIONS)}) trovato nella directory: {image_dir_path}")
        return True 

    logger.info(f"--- Riepilogo Elaborazione Directory ---")
    logger.info(f"Directory: {image_dir_path}")
    logger.info(f"File totali elaborati (con estensione supportata): {processed_files_count}")
    logger.info(f"Trascrizioni riuscite: {successful_files_count}")
    logger.info(f"Trascrizioni fallite: {failed_files_count}")
    
    if failed_files_count > 0:
        logger.warning("Alcune trascrizioni sono fallite durante l'elaborazione della directory.")
        return False
    return True

def main():
    """
    Main function to parse arguments and orchestrate the transcription process.
    """
    logger.info("Avvio Transcription Tool CLI")
    
    # OPENAI_API_CONFIGURED check is now inside process_single_image, 
    # as local classification can run without it.
    # However, we might still want an early exit if the ultimate goal (transcription) is impossible.
    # For now, let's rely on the check within process_single_image/process_directory.
    # If a large batch is run, failing early if OpenAI isn't set up might be better.
    # Consider adding a check here if args.image_file or args.image_dir is provided
    # and OPENAI_API_CONFIGURED is False, then exit.
    # For now, the current logic is fine as it will fail at the first transcription attempt.

    parser = create_parser() # Use the refactored parser creation
    args = parser.parse_args()
    logger.debug(f"Argomenti CLI ricevuti: {args}")

    if args.output_dir:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory di output specificata e assicurata/creata: {args.output_dir.resolve()}")
        except PermissionError:
            logger.error(f"Permesso negato nella creazione della directory di output specificata: {args.output_dir.resolve()}", exc_info=True)
            sys.exit(1)
        except OSError as e:
            logger.error(f"Errore OS durante la creazione della directory di output {args.output_dir.resolve()}: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Errore imprevisto durante la creazione della directory di output {args.output_dir.resolve()}: {e}", exc_info=True)
            sys.exit(1)

    prompt_text = get_prompt_content(args.prompt_file)
    if prompt_text is None:
        logger.critical("Impossibile caricare il prompt. Terminazione del programma.")
        sys.exit(1) 

    overall_success = False
    if args.image_file:
        if not args.image_file.is_file():
            logger.error(f"Il file immagine specificato non esiste o non è un file: {args.image_file}")
            sys.exit(1)
        overall_success = process_single_image(args.image_file, prompt_text, args.output_dir)
    elif args.image_dir:
        if not args.image_dir.is_dir():
            logger.error(f"La directory immagini specificata non esiste o non è una directory: {args.image_dir}")
            sys.exit(1)
        overall_success = process_directory(args.image_dir, prompt_text, args.output_dir)

    if overall_success:
        logger.info("Elaborazione completata con successo.")
        sys.exit(0) 
    else:
        logger.error("Elaborazione completata con uno o più errori. Controllare i log per i dettagli.")
        sys.exit(1) 

if __name__ == "__main__":
    # This structure allows the script to be run directly (python src/transcription_tool.py)
    # or as a module (python -m src.transcription_tool)
    main() 