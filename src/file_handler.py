from pathlib import Path
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg"]

def validate_image_file(image_path: Path) -> bool:
    """
    Validates if the given path points to a supported image file.

    Checks for file existence, if it's actually a file (not a directory),
    and if its extension is one of the `SUPPORTED_IMAGE_EXTENSIONS` (case-insensitive).

    Args:
        image_path (Path): The path to the image file to validate.

    Returns:
        bool: True if the image file is valid, False otherwise.
    """
    if not isinstance(image_path, Path):
        logger.error(f"Percorso non valido fornito a validate_image_file: {image_path} (tipo: {type(image_path)}). Previsto un oggetto Path.")
        return False

    if not image_path.exists():
        logger.warning(f"File immagine non trovato: {image_path}")
        return False
    if not image_path.is_file():
        logger.warning(f"Il percorso immagine specificato non è un file: {image_path}")
        return False
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logger.warning(
            f"Estensione file non supportata per {image_path.name}. "
            f"Supportate: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)} (ignora maiuscole/minuscole)."
        )
        return False
    logger.debug(f"File immagine valido: {image_path}")
    return True

def save_transcription(
    transcription_text: str,
    input_image_path: Path,
    output_directory: Optional[Path] = None
) -> Optional[Path]:
    """
    Saves the transcription text to a .txt file.

    The output file will be named after the input image file (e.g., image.jpg -> image.txt).
    If `output_directory` is not specified, it defaults to the directory of the input image.
    If `output_directory` is specified, it will be created if it doesn't exist.

    Args:
        transcription_text (str): The text to save.
        input_image_path (Path): Path to the original image file, used for naming the output.
        output_directory (Path | None, optional): Directory to save the .txt file.
                                                 Defaults to the input image's directory.

    Returns:
        Path | None: The path to the saved .txt file, or None if an error occurred.
    """
    if not isinstance(input_image_path, Path):
        logger.error(f"Percorso immagine di input non valido: {input_image_path}. Previsto Path.")
        return None

    if output_directory is None:
        output_dir = input_image_path.parent
        logger.debug(f"Nessuna directory di output specificata, utilizzo: {output_dir}")
    else:
        output_dir = output_directory
        if not isinstance(output_dir, Path):
            logger.error(f"Percorso directory di output non valido: {output_dir}. Previsto Path.")
            return None
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory di output assicurata/creata: {output_dir}")
        except PermissionError:
            logger.error(f"Permesso negato nella creazione della directory di output: {output_dir}", exc_info=True)
            return None
        except OSError as e:
            logger.error(f"Errore OS durante la creazione della directory di output {output_dir}: {e}", exc_info=True)
            return None
        except Exception as e: # Catch any other mkdir related error
            logger.error(f"Errore imprevisto durante la creazione della directory di output {output_dir}: {e}", exc_info=True)
            return None

    # Create the output file name (e.g., image.jpg -> image.txt)
    output_filename = input_image_path.stem + ".txt"
    output_file_path = output_dir / output_filename

    try:
        with open(output_file_path, "w", encoding='utf-8') as f:
            f.write(transcription_text)
        logger.info(f"Trascrizione salvata con successo in: {output_file_path}")
        return output_file_path
    except IOError as e:
        logger.error(f"Errore I/O durante il salvataggio della trascrizione in {output_file_path}: {e}", exc_info=True)
        return None
    except PermissionError:
        logger.error(f"Permesso negato durante il salvataggio della trascrizione in {output_file_path}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Errore imprevisto durante il salvataggio della trascrizione in {output_file_path}: {e}", exc_info=True)
        return None

def load_prompt_file(prompt_file_path: Path) -> Optional[str]:
    """
    Loads the content of a text file, typically a prompt for the LLM.

    Args:
        prompt_file_path (Path): The path to the prompt text file.

    Returns:
        str | None: The content of the file as a string, or None if an error occurs
                    (e.g., file not found, permission denied, decoding error).
    """
    if not isinstance(prompt_file_path, Path):
        logger.error(f"Percorso file prompt non valido: {prompt_file_path}. Previsto Path.")
        return None

    if not prompt_file_path.is_file():
        logger.error(f"File prompt non trovato o non è un file: {prompt_file_path}")
        return None

    try:
        with open(prompt_file_path, "r", encoding='utf-8') as f:
            prompt_content = f.read().strip()
        if not prompt_content:
            logger.warning(f"Il file prompt {prompt_file_path} è vuoto.")
            # Decide if an empty prompt is an error or acceptable
            # For now, returning it as is, but could return None or raise error
        logger.info(f"Contenuto del prompt caricato con successo da: {prompt_file_path}")
        logger.debug(f"Contenuto prompt (prime 100 chars): {prompt_content[:100]}...")
        return prompt_content
    except FileNotFoundError: # Should be caught by is_file(), but for robustness
        logger.error(f"File prompt non trovato (inner try): {prompt_file_path}")
        return None
    except PermissionError:
        logger.error(f"Permesso negato durante la lettura del file prompt: {prompt_file_path}", exc_info=True)
        return None
    except UnicodeDecodeError as e:
        logger.error(f"Errore di decodifica durante la lettura del file prompt {prompt_file_path} (assicurarsi che sia UTF-8): {e}", exc_info=True)
        return None
    except IOError as e:
        logger.error(f"Errore I/O durante la lettura del file prompt {prompt_file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Errore imprevisto durante la lettura del file prompt {prompt_file_path}: {e}", exc_info=True)
        return None 