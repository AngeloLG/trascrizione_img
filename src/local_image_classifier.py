from pathlib import Path
import logging
from typing import Optional

from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification # Standard imports for image classification
import torch

logger = logging.getLogger(__name__)

# Updated model name as per user request
MODEL_NAME = "microsoft/dit-base-finetuned-rvlcdip"

IMAGE_CLASSIFIER = None # Standard name for an image classifier pipeline
PROCESSOR = None
MODEL = None

# Heuristic threshold for confidence score (optional, can be adjusted)
MIN_CONFIDENCE_SCORE = 0.5 

def _initialize_classifier(): # Renamed back to generic classifier init
    """Initializes the Hugging Face DiT model and processor for image classification."""
    global IMAGE_CLASSIFIER, PROCESSOR, MODEL
    
    if IMAGE_CLASSIFIER is None:
        try:
            logger.info(f"Inizializzazione del classificatore di immagini locale con il modello: {MODEL_NAME}")
            
            PROCESSOR = AutoImageProcessor.from_pretrained(MODEL_NAME)
            MODEL = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
            IMAGE_CLASSIFIER = pipeline("image-classification", model=MODEL, feature_extractor=PROCESSOR)

            logger.info(f"Classificatore di immagini DiT ({MODEL_NAME}) inizializzato con successo.")
        except Exception as e:
            logger.error(f"Errore durante l'inizializzazione del classificatore DiT ({MODEL_NAME}): {e}", exc_info=True)
            logger.error("Assicurarsi che il modello DiT sia corretto e che le dipendenze siano installate.")
            IMAGE_CLASSIFIER = None 
            PROCESSOR = None
            MODEL = None

_initialize_classifier() # Initialize when module is loaded

def classify_text_type_local(image_path: Path) -> tuple[Optional[str], Optional[list[dict]]]:
    """
    Classifies the image content primarily as 'handwritten', 'typewritten', or 'other_document_type'
    using a local DiT model fine-tuned on RVL-CDIP. 
    Also returns all predictions from the model for further analysis.

    Args:
        image_path (Path): The path to the image file.

    Returns:
        tuple[str | None, list[dict] | None]: 
            - Primary classification ("handwritten", "typewritten", "other_document_type", or "undetermined").
            - List of all prediction dictionaries (e.g., [{'label': 'letter', 'score': 0.9}, ...]) from DiT, or None.
            None for primary classification if a critical error occurs.
    """
    if IMAGE_CLASSIFIER is None or PROCESSOR is None or MODEL is None:
        logger.error("Classificatore di immagini DiT non inizializzato. Impossibile classificare.")
        return None, None

    try:
        img = Image.open(image_path).convert("RGB") 
        logger.info(f"Classificazione DiT in corso per l'immagine: {image_path.name} con {MODEL_NAME}")
        
        # Get all predictions from the classifier
        # The pipeline by default might only return the top_k=1, ensure we get more if needed
        # For DiT with RVL-CDIP, it's trained on 16 classes. We can inspect the top few.
        predictions = IMAGE_CLASSIFIER(img, top_k=5) # Get top 5 predictions
        
        if not predictions or not isinstance(predictions, list):
            logger.warning(f"Nessuna predizione o output imprevisto da DiT per {image_path.name}.")
            logger.debug(f"Output grezzo da DiT per {image_path.name}: {predictions}")
            return "undetermined", None
        
        logger.debug(f"Predizioni DiT per {image_path.name}: {predictions}")

        # --- Label Mapping Logic for DiT RVL-CDIP --- 
        # RVL-CDIP classes: letter, form, email, handwritten, advertisement, 
        # scientific report, scientific publication, specification, file folder, 
        # news article, budget, invoice, presentation, questionnaire, resume, memo

        primary_classification = "undetermined"
        top_label = ""
        top_score = 0.0

        if predictions and predictions[0]:
            top_prediction = predictions[0]
            top_label = top_prediction['label'].lower()
            top_score = top_prediction['score']
            logger.info(f"Predizione DiT principale per {image_path.name}: label='{top_label}', score={top_score:.4f}")

            if top_score < MIN_CONFIDENCE_SCORE:
                logger.info(f"Punteggio di confidenza ({top_score:.4f}) per '{top_label}' è inferiore alla soglia ({MIN_CONFIDENCE_SCORE}). Classificato come 'undetermined'.")
                primary_classification = "undetermined"
            elif top_label == "handwritten":
                primary_classification = "handwritten"
            elif top_label in ["letter", "form", "email", "memo", "resume", "scientific publication", "report", "specification", "news article", "invoice"]:
                # These are typically typewritten or primarily typewritten
                primary_classification = "typewritten"
            elif top_label in ["advertisement", "budget", "file folder", "presentation", "questionnaire", "scientific report"]:
                # These could be mixed, or their primary characteristic isn't just text style for our purpose
                primary_classification = "other_document_type"
            else:
                logger.info(f"Etichetta DiT principale '{top_label}' non mappata esplicitamente. Considerata 'other_document_type' se confidenza alta, altrimenti 'undetermined'.")
                primary_classification = "other_document_type" # Or "undetermined" if preferred for unmapped high-confidence
        
        logger.info(f"Classificazione primaria DiT per {image_path.name}: '{primary_classification}'")
        return primary_classification, predictions # Return all predictions as well

    except FileNotFoundError:
        logger.error(f"File immagine non trovato per la classificazione DiT: {image_path}")
        return None, None
    except Exception as e:
        logger.error(f"Errore durante la classificazione DiT sull'immagine {image_path.name}: {e}", exc_info=True)
        return None, None 