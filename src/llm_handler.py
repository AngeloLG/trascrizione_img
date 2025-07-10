from pathlib import Path
import logging
import os
import base64
from openai import OpenAI, APIError, AuthenticationError, RateLimitError, NotFoundError, BadRequestError # Import specific OpenAI errors
from PIL import Image # For loading images
from dotenv import load_dotenv # Import load_dotenv
from typing import Optional # Import Optional
from transformers import TrOCRProcessor, VisionEncoderDecoderModel # Removed DonutProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global OpenAI client instance
client: Optional[OpenAI] = None
OPENAI_API_CONFIGURED = False

# --- Local OCR Transcription for Handwritten Text ---
LOCAL_OCR_PROCESSOR = None
LOCAL_OCR_MODEL = None
LOCAL_OCR_MODEL_NAME = "microsoft/trocr-base-handwritten"

def configure_openai_api() -> bool:
    """
    Configures the OpenAI API client using the API key from environment variables.

    This function attempts to load a .env file from the project root first,
    then retrieves the OPENAI_API_KEY. It initializes the global `client`
    if the key is found and the client can be instantiated.

    Returns:
        bool: True if the OpenAI client was successfully configured, False otherwise.
    """
    global client, OPENAI_API_CONFIGURED
    
    if OPENAI_API_CONFIGURED and client is not None:
        logger.debug("OpenAI client already configured.")
        return True

    # Load environment variables from .env file if present
    env_path = Path('.').resolve() / '.env'
    dotenv_loaded = False
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Variabili d'ambiente caricate da: {env_path}")
        dotenv_loaded = True
    else:
        logger.info("File .env non trovato nella root del progetto. Si utilizzeranno le variabili d'ambiente di sistema, se presenti.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "La variabile d'ambiente OPENAI_API_KEY non è stata trovata. "
            "Assicurarsi che sia definita nel file .env nella root del progetto "
            "o come variabile d'ambiente di sistema."
        )
        logger.info("È possibile ottenere una chiave API OpenAI da: https://platform.openai.com/api-keys")
        OPENAI_API_CONFIGURED = False
        return False
    
    if dotenv_loaded:
        logger.info("OPENAI_API_KEY caricata da file .env.")
    else:
        logger.info("OPENAI_API_KEY caricata da variabili d'ambiente di sistema.")

    try:
        client = OpenAI(api_key=api_key)
        logger.info("Client OpenAI API configurato con successo.")
        OPENAI_API_CONFIGURED = True
        return True
    except Exception as e:
        logger.error(f"Errore imprevisto durante la configurazione del client OpenAI API: {e}", exc_info=True)
        OPENAI_API_CONFIGURED = False
        return False

# Attempt to configure the API when the module is loaded.
if not OPENAI_API_CONFIGURED:
    configure_openai_api()

def image_to_base64(image_path: Path) -> Optional[str]:
    """
    Converts an image file to a base64 encoded string with a data URI prefix.

    Args:
        image_path (Path): The path to the image file.

    Returns:
        str | None: The base64 encoded string of the image (e.g., "data:image/jpeg;base64,..."),
                    or None if an error occurs (e.g., file not found, processing error).
    """
    if not image_path.is_file():
        logger.error(f"File immagine non trovato (image_to_base64): {image_path}")
        return None

    try:
        with Image.open(image_path) as img:
            img_format = img.format
        
        if not img_format:
            logger.warning(
                f"Formato immagine non rilevato automaticamente per {image_path.name}. "
                f"Tentativo di deduzione dall'estensione del file. Altrimenti, si userà 'jpeg'."
            )
            # Try to guess from extension
            ext = image_path.suffix.lower().replace('.', '')
            if ext in ['jpeg', 'jpg', 'png', 'gif', 'bmp', 'webp']:
                img_format = ext
                if ext == 'jpg': # common practice
                    img_format = 'jpeg'
            else:
                img_format = "jpeg" # Default fallback
            logger.info(f"Formato immagine impostato a '{img_format}' per {image_path.name}.")
        else:
            img_format = img_format.lower()
            if img_format == 'jpg': # common practice
                img_format = 'jpeg'


        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        data_uri = f"data:image/{img_format};base64,{encoded_string}"
        logger.debug(f"Immagine {image_path.name} convertita in base64 ({len(data_uri)} caratteri).")
        return data_uri
    except FileNotFoundError: # Should be caught by is_file() check, but good for robustness
        logger.error(f"File immagine non trovato (image_to_base64, inner try): {image_path}")
        return None
    except IOError as e:
        logger.error(f"Errore I/O durante l'apertura o lettura dell'immagine {image_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Errore imprevisto durante la conversione dell'immagine {image_path} in base64: {e}", exc_info=True)
        return None

def get_transcription_from_llm(image_path: Path, prompt_text: str) -> Optional[str]:
    """
    Retrieves text transcription for a given image using the configured OpenAI model.

    This function encodes the image to base64, then sends it along with the
    prompt to the OpenAI API. It handles various API errors and logs them.

    Args:
        image_path (Path): The path to the image file.
        prompt_text (str): The prompt text to guide the LLM's transcription.

    Returns:
        str | None: The transcribed text from OpenAI, or None if any error occurs
                    during API configuration, image processing, or the API call itself.
    """
    global client, OPENAI_API_CONFIGURED
    
    if not OPENAI_API_CONFIGURED or client is None:
        logger.warning("Client OpenAI API non configurato al momento della chiamata a get_transcription_from_llm. Tentativo di riconfigurazione...")
        if not configure_openai_api():
            logger.error("Fallita riconfigurazione del client OpenAI API. Impossibile procedere con la trascrizione.")
            return None
        # If configure_openai_api() was successful, client is now initialized.
        logger.info("Riconfigurazione del client OpenAI API riuscita.")

    logger.info(f"Tentativo di trascrizione per l'immagine: {image_path.name} con prompt.")
    logger.debug(f"Prompt utilizzato (prime 100 chars): {prompt_text[:100]}...")

    base64_image = image_to_base64(image_path)
    if not base64_image:
        logger.error(f"Fallita conversione in base64 per l'immagine {image_path.name}. Impossibile procedere.")
        return None

    # WARNING: "gpt-4o" is not a standard OpenAI vision model.
    # If this is not a special model with vision capabilities,
    # image processing will likely fail.
    model_to_use = "gpt-4.1-mini"
    logger.info(f"Utilizzo del modello OpenAI: {model_to_use}")

    try:
        logger.debug(f"Invio richiesta a OpenAI API per {image_path.name} con modello {model_to_use}.")
        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        },
                    ],
                }
            ],
            max_tokens=1500 # Adjust as needed
        )
        
        logger.debug(f"Risposta grezza da OpenAI API per {image_path.name}: {response}")

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            transcription = response.choices[0].message.content.strip()
            logger.info(f"Trascrizione ricevuta con successo da OpenAI per {image_path.name}.")
            logger.debug(f"Trascrizione completa per {image_path.name}: {transcription}")
            return transcription
        else:
            logger.warning(f"La risposta da OpenAI per {image_path.name} non contiene il contenuto atteso o è vuota.")
            if response.choices and response.choices[0].finish_reason:
                logger.warning(f"Motivo di terminazione della risposta OpenAI: {response.choices[0].finish_reason}")
            else:
                logger.warning("Nessun messaggio o contenuto nella prima scelta della risposta OpenAI.")
            logger.debug(f"Dettagli risposta problematica: {response}")
            return None

    except AuthenticationError as e:
        logger.error(f"Errore di autenticazione con OpenAI API per {image_path.name}: {e}")
        logger.error("Verificare la validità e i permessi della OPENAI_API_KEY.")
        return None
    except RateLimitError as e:
        logger.error(f"Rate limit superato con OpenAI API per {image_path.name}: {e}")
        logger.error("Attendere prima di effettuare nuove richieste o controllare i limiti del proprio piano.")
        return None
    except NotFoundError as e:
        logger.error(f"Errore NotFound (404) da OpenAI API per {image_path.name}: {e}")
        logger.error(f"Potrebbe indicare un modello non valido ('{model_to_use}') o un endpoint errato.")
        return None
    except BadRequestError as e: # Often for issues with the request payload itself
        logger.error(f"Errore BadRequest (400) da OpenAI API per {image_path.name}: {e}")
        logger.error("Questo può essere dovuto a un formato immagine non supportato, un prompt malformato, o il modello non supporta l'input fornito (es. immagini).")
        logger.debug(f"Dettagli errore BadRequest: {e.response.text if e.response else 'N/A'}")
        return None
    except APIError as e: # Catch-all for other API related errors from OpenAI
        logger.error(f"Errore generico API da OpenAI per {image_path.name}: {e}", exc_info=True)
        if hasattr(e, 'status_code'):
            logger.error(f"Status code OpenAI: {e.status_code}")
        if hasattr(e, 'code'):
            logger.error(f"Codice errore OpenAI: {e.code}")
        return None
    except Exception as e: # Catch unexpected errors not specific to OpenAI's library
        logger.error(f"Errore imprevisto durante la chiamata API a OpenAI per {image_path.name}: {e}", exc_info=True)
        return None 

# --- Funzione di classificazione basata su OpenAI (rimossa/commentata) ---
# def classify_image_type(image_path: Path) -> Optional[str]:
#     """
#     Classifies the image content as 'handwritten', 'typewritten', or 'undetermined'.
#
#     Uses the configured OpenAI vision model to analyze the image.
#
#     Args:
#         image_path (Path): The path to the image file.
#
#     Returns:
#         str | None: The classified image type ("handwritten", "typewritten", "undetermined"),
#                     or None if any error occurs.
#     """
#     global client, OPENAI_API_CONFIGURED
#
#     if not OPENAI_API_CONFIGURED or client is None:
#         logger.warning("Client OpenAI API non configurato al momento della chiamata a classify_image_type. Tentativo di riconfigurazione...")
#         if not configure_openai_api():
#             logger.error("Fallita riconfigurazione del client OpenAI API. Impossibile procedere con la classificazione.")
#             return None
#         logger.info("Riconfigurazione del client OpenAI API riuscita.")
#
#     logger.info(f"Tentativo di classificazione tipo testo per l'immagine: {image_path.name}")
#
#     base64_image = image_to_base64(image_path)
#     if not base64_image:
#         logger.error(f"Fallita conversione in base64 per l'immagine {image_path.name}. Impossibile procedere con la classificazione.")
#         return None
#
#     model_to_use = "gpt-4o" 
#     classification_prompt = (
#         "Analyze the provided image and determine if the primary text content is "
#         "predominantly handwritten or typewritten. "
#         "Respond with a single word: \"handwritten\", \"typewritten\", or \"undetermined\" "
#         "if it's a mix, unclear, or no text is present."
#     )
#     
#     valid_classifications = ["handwritten", "typewritten", "undetermined"]
#
#     try:
#         logger.debug(f"Invio richiesta classificazione a OpenAI API per {image_path.name} con modello {model_to_use}.")
#         response = client.chat.completions.create(
#             model=model_to_use,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": classification_prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {"url": base64_image},
#                         },
#                     ],
#                 }
#             ],
#             max_tokens=50 # Small response expected
#         )
#         
#         logger.debug(f"Risposta grezza da OpenAI API per classificazione {image_path.name}: {response}")
#
#         if response.choices and response.choices[0].message and response.choices[0].message.content:
#             classification_result = response.choices[0].message.content.strip().lower()
#             logger.info(f"Classificazione ricevuta da OpenAI per {image_path.name}: '{classification_result}'")
#             
#             if classification_result in valid_classifications:
#                 return classification_result
#             else:
#                 logger.warning(f"Classificazione '{classification_result}' non è una delle attese ({', '.join(valid_classifications)}). Considerata 'undetermined'.")
#                 return "undetermined"
#         else:
#             logger.warning(f"La risposta da OpenAI per classificazione {image_path.name} non contiene il contenuto atteso o è vuota.")
#             if response.choices and response.choices[0].finish_reason:
#                 logger.warning(f"Motivo di terminazione della risposta OpenAI (classificazione): {response.choices[0].finish_reason}")
#             return "undetermined"
#
#     except AuthenticationError as e:
#         logger.error(f"Errore di autenticazione con OpenAI API (classificazione) per {image_path.name}: {e}")
#         return None
#     except RateLimitError as e:
#         logger.error(f"Rate limit superato con OpenAI API (classificazione) per {image_path.name}: {e}")
#         return None
#     except NotFoundError as e:
#         logger.error(f"Errore NotFound (404) da OpenAI API (classificazione) per {image_path.name} (modello: '{model_to_use}'): {e}")
#         return None
#     except BadRequestError as e:
#         logger.error(f"Errore BadRequest (400) da OpenAI API (classificazione) per {image_path.name}: {e}")
#         logger.debug(f"Dettagli errore BadRequest (classificazione): {e.response.text if e.response else 'N/A'}")
#         return None
#     except APIError as e:
#         logger.error(f"Errore generico API da OpenAI (classificazione) per {image_path.name}: {e}", exc_info=True)
#         return None
#     except Exception as e:
#         logger.error(f"Errore imprevisto durante la chiamata API per classificazione {image_path.name}: {e}", exc_info=True)
#         return None 

def _initialize_local_ocr_model() -> bool:
    """
    Initializes the local OCR model and processor (TrOCR).

    Returns:
        bool: True if initialization is successful, False otherwise.
    """
    global LOCAL_OCR_PROCESSOR, LOCAL_OCR_MODEL
    if LOCAL_OCR_PROCESSOR and LOCAL_OCR_MODEL:
        logger.debug(f"Modello OCR locale ({LOCAL_OCR_MODEL_NAME}) già inizializzato.")
        return True

    try:
        logger.info(f"Inizializzazione del modello OCR locale: {LOCAL_OCR_MODEL_NAME}. Potrebbe richiedere tempo al primo avvio per il download...")
        # Use TrOCRProcessor
        LOCAL_OCR_PROCESSOR = TrOCRProcessor.from_pretrained(LOCAL_OCR_MODEL_NAME)
        LOCAL_OCR_MODEL = VisionEncoderDecoderModel.from_pretrained(LOCAL_OCR_MODEL_NAME)
        logger.info(f"Modello OCR locale ({LOCAL_OCR_MODEL_NAME}) e processore ({LOCAL_OCR_PROCESSOR.__class__.__name__}) inizializzati con successo.")
        return True
    except Exception as e:
        logger.error(f"Fallimento durante l'inizializzazione del modello OCR locale ({LOCAL_OCR_MODEL_NAME}): {e}", exc_info=True)
        LOCAL_OCR_PROCESSOR = None
        LOCAL_OCR_MODEL = None
        return False

def get_transcription_from_local_ocr(image_path: Path) -> Optional[str]:
    """
    Retrieves text transcription for a given image using a local TrOCR model.

    Args:
        image_path (Path): The path to the image file.

    Returns:
        str | None: The transcribed text, or None if any error occurs.
    """
    # Rimuoviamo il try-except esterno per vedere errori grezzi se ci sono
    # try:
    print(f"DEBUG PRINT: ENTERING get_transcription_from_local_ocr for {image_path.name}", flush=True)
    logger.debug(f"ENTERING get_transcription_from_local_ocr for {image_path.name} - THIS IS A DEBUG TEST LOG")
    global LOCAL_OCR_PROCESSOR, LOCAL_OCR_MODEL

    print("DEBUG PRINT: Before _initialize_local_ocr_model check", flush=True)
    if not _initialize_local_ocr_model() or not LOCAL_OCR_PROCESSOR or not LOCAL_OCR_MODEL:
        logger.error(f"Modello OCR locale ({LOCAL_OCR_MODEL_NAME}) non inizializzato correttamente. Impossibile procedere con la trascrizione locale.")
        print("DEBUG PRINT: Model not initialized, returning None", flush=True)
        return None
    print("DEBUG PRINT: After _initialize_local_ocr_model check, model appears initialized.", flush=True)

    logger.info(f"Tentativo di trascrizione locale (OCR) per l'immagine: {image_path.name} utilizzando {LOCAL_OCR_MODEL_NAME}.")
    print(f"DEBUG PRINT: Attempting local OCR for {image_path.name} with {LOCAL_OCR_MODEL_NAME}", flush=True)

    try:
        print("DEBUG PRINT: Inside main try block, before Image.open", flush=True)
        image = Image.open(image_path).convert("RGB")
        print(f"DEBUG PRINT: Image {image_path.name} loaded. Size: {image.size}", flush=True)
        logger.debug(f"Immagine {image_path.name} caricata e convertita in RGB. Dimensioni: {image.size}")
        
        print("DEBUG PRINT: Before LOCAL_OCR_PROCESSOR call", flush=True)
        pixel_values = LOCAL_OCR_PROCESSOR(images=image, return_tensors="pt").pixel_values
        print(f"DEBUG PRINT: After LOCAL_OCR_PROCESSOR call. Shape: {pixel_values.shape}", flush=True)
        logger.debug(f"Immagine {image_path.name} processata dal TrOCRProcessor. Shape dei pixel_values: {pixel_values.shape}")

        print("DEBUG PRINT: Before LOCAL_OCR_MODEL.generate call", flush=True)
        logger.debug(f"Chiamata a LOCAL_OCR_MODEL.generate() con i parametri di default per {image_path.name}.")
        generated_ids = LOCAL_OCR_MODEL.generate(pixel_values)
        print(f"DEBUG PRINT: After LOCAL_OCR_MODEL.generate call. Generated IDs: {generated_ids}", flush=True)
        logger.debug(f"ID generati da TrOCR per {image_path.name}: {generated_ids}")

        print("DEBUG PRINT: Before batch_decode call", flush=True)
        transcription = LOCAL_OCR_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"DEBUG PRINT: After batch_decode call. Transcription: '{transcription}'", flush=True)
        logger.info(f"Trascrizione locale (OCR) ricevuta con successo per {image_path.name}.")
        logger.debug(f"Trascrizione locale completa per {image_path.name}: {transcription}")
        return transcription.strip()

    except FileNotFoundError:
        logger.error(f"File immagine non trovato per OCR locale (inner try): {image_path}")
        print(f"DEBUG PRINT: FileNotFoundError for {image_path}", flush=True)
        return None
    except IOError as e_io:
        logger.error(f"Errore I/O durante l\'apertura o lettura dell\'immagine {image_path} per OCR locale (inner try): {e_io}", exc_info=True)
        print(f"DEBUG PRINT: IOError for {image_path}: {e_io}", flush=True)
        return None
    except Exception as e_ocr:
        logger.error(f"Errore imprevisto durante la trascrizione OCR locale (inner try) per {image_path.name}: {e_ocr}", exc_info=True)
        print(f"DEBUG PRINT: Exception in OCR block for {image_path.name}: {e_ocr}", flush=True)
        return None
            
    # except Exception as e_outer: # Rimosso per ora per vedere errori grezzi
    #     logger.error(f"ECCEZIONE ESTERNA CATTURATA in get_transcription_from_local_ocr per {image_path.name}: {e_outer}", exc_info=True)
    #     return None 