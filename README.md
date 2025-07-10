# Tool di Trascrizione Immagini JPG con LLM (OpenAI)

Questo strumento a riga di comando (CLI) permette di trascrivere il testo contenuto in immagini JPG (o JPEG) utilizzando un modello linguistico di grandi dimensioni (LLM) fornito da OpenAI. È possibile processare un singolo file immagine o un'intera directory di immagini.

## Funzionalità Principali

- Trascrizione di testo da file immagine `.jpg` o `.jpeg`.
- Supporto per l'elaborazione di un singolo file immagine o di tutti i file immagine supportati in una directory.
- **Classificazione preliminare dell'immagine eseguita localmente** (utilizzando il modello `microsoft/dit-base-finetuned-rvlcdip` da Hugging Face Transformers) per determinare il tipo di documento (es. "handwritten", "typewritten", "other_document_type", "undetermined") basandosi sulle classi RVL-CDIP. Questo evita costi API per la classificazione.
- **Trascrizione OCR Locale Condizionale (Attualmente Sospesa/In Revisione):**
  - La funzionalità che prevedeva l'uso del modello OCR locale `microsoft/trocr-base-handwritten` per le immagini classificate come "handwritten" è **attualmente sospesa** in attesa di identificare un modello più performante per questa specifica tipologia di documenti.
  - Allo stato attuale, tutte le immagini, indipendentemente dalla classificazione DiT (inclusa "handwritten"), vengono inviate al modello LLM OpenAI per la trascrizione, se non diversamente specificato da future modifiche.
- Utilizzo di un prompt personalizzabile per guidare il processo di trascrizione del LLM (quando si usa OpenAI).
- Possibilità di specificare un prompt di default se non ne viene fornito uno dall'utente (`default_transcription_prompt.txt`). Questo prompt è ora ottimizzato per documenti **dattiloscritti in italiano della prima metà del XX secolo.**
- Salvataggio delle trascrizioni in file di testo (`.txt`) separati, nominati in base al file immagine originale.
- Configurazione della directory di output per i file di trascrizione.
- Gestione sicura della chiave API OpenAI tramite file `.env`.
- Logging dettagliato delle operazioni su console e su file (`logs/transcription_tool.log`) per il debug e il monitoraggio.
- Gestione robusta degli errori e codici di uscita appropriati.

## Prerequisiti

- Python 3.8 o superiore.
- Una chiave API OpenAI valida. Potete ottenerne una da [OpenAI Platform](https://platform.openai.com/api-keys).
- Git (per clonare il repository, se applicabile).

## Installazione

1.  **Clonare il Repository (Opzionale):**
    Se avete scaricato il codice sorgente come archivio, decomprimetelo. Altrimenti, potete clonare il repository (se disponibile):

    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Creare un Ambiente Virtuale (Raccomandato):**
    È buona norma utilizzare un ambiente virtuale per isolare le dipendenze del progetto.

    ```bash
    python -m venv venv
    ```

    Attivare l'ambiente virtuale:

    - Su Windows:
      ```bash
      venv\Scripts\activate
      ```
    - Su macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

3.  **Installare le Dipendenze:**
    Assicurarsi che il file `requirements.txt` sia presente nella directory principale del progetto. Quindi, eseguire:
    ```bash
    pip install -r requirements.txt
    ```
    Questo installerà `openai`, `Pillow`, `python-dotenv`, `transformers`, e `torch` (o un altro backend per Hugging Face se configurato diversamente).
    Nota: il download dei modelli Hugging Face (`microsoft/dit-base-finetuned-rvlcdip` per la classificazione e `microsoft/trocr-base-handwritten` per l'OCR locale, se necessario) avverrà al primo utilizzo.

## Configurazione

1.  **Chiave API OpenAI:**
    Questo tool richiede una chiave API OpenAI per funzionare. La chiave deve essere memorizzata in un file `.env` situato nella directory principale del progetto.

    - **Creare il file `.env.example**:
      Se non esiste già, create un file chiamato `.env.example` (o `dotenv_template.txt` se avete seguito i passi precedenti) nella root del progetto con il seguente contenuto:
      ```
      OPENAI_API_KEY="LA_TUA_CHIAVE_API_QUI"
      ```
    - **Creare il file `.env**:
      Copiate `.env.example` (o `dotenv_template.txt`) in un nuovo file chiamato `.env` nella stessa directory:
      ```bash
      cp .env.example .env
      # oppure (su Windows)
      # copy dotenv_template.txt .env
      ```
      Aprite il file `.env` e sostituite `"LA_TUA_CHIAVE_API_QUI"` con la vostra chiave API OpenAI effettiva.
      **Importante:** Il file `.env` contiene informazioni sensibili e non dovrebbe essere committato nel version control (è già incluso nel `.gitignore`).

2.  **Prompt di Default:**
    Il tool utilizza un prompt di default se non ne viene specificato uno dall'utente. Questo prompt si trova nel file `default_transcription_prompt.txt` nella directory principale del progetto. Potete modificare il contenuto di questo file per cambiare il prompt di default.
    Il prompt di default fornito è ora specificamente pensato per documenti dattiloscritti della prima metà del '900 e non fa più riferimenti generici a testo manoscritto:

    ```text
    ## Task

    Transcribe the textual content visible in the attached JPG image.

    ## Language context
    The document is in **Italian**, is **typewritten**, and dates from the **first half of the 20th century**. It may include personal names, place names, and idiomatic expressions typical of the period.

    ## Instructions

    - Carefully **analyze the image visually**: do **not** use OCR scripts.
    - Recognize the **typewritten text**.
    - If the text is **legible**, transcribe it **faithfully**, preserving:
      - original **spelling**, **punctuation**, and **line breaks**;
      - marginal notes, annotations, corrections or crossed-out text (e.g., use ~~strikethrough~~ if text is clearly struck through);
      - layout approximations (indentations, paragraph divisions, spacing).

    - If the text is **partially unreadable**, transcribe the **readable portions** and indicate gaps with `[...]`.
    - If **no legible text** is present, write:
      `No legible text found in the image. No transcription possible.`

    - Do **not** summarize, interpret, or translate the content.
    - The output must be a **neutral, complete, and accurate** transcription.

    ## Input
    <input-image>
    {image.jpg}
    </input-image>

    ## Output
    A faithful transcription of all legible text in Italian, or:
    `No legible text found in the image. No transcription possible.`
    ```

3.  **Modello LLM Configurato:**
    Attualmente, il tool è configurato per utilizzare il modello OpenAI `\"gpt-4.1-mini\"` per le trascrizioni non gestite dall\'OCR locale.
    **ATTENZIONE:** Questo NON è un modello OpenAI standard noto per le capacità di visione. Se questo specifico alias di modello non è disponibile per il vostro account o non supporta l'input di immagini, le trascrizioni che richiedono OpenAI falliranno. Per l'elaborazione di immagini, modelli come `gpt-4-turbo` (con l'ultima versione che include vision) o `gpt-4-vision-preview` sono più comunemente utilizzati. Potrebbe essere necessario modificare `model_to_use` in `src/llm_handler.py` se si riscontrano problemi con le trascrizioni via OpenAI.

## Utilizzo

Eseguire lo script dalla directory principale del progetto utilizzando `python -m src.transcription_tool` seguito dagli argomenti necessari.

**Argomenti della Riga di Comando:**

- `--image_file PERCORSO_FILE_IMMAGINE`: Specifica il percorso di un singolo file immagine JPG/JPEG da processare.
- `--image_dir PERCORSO_DIRECTORY_IMMAGINI`: Specifica il percorso di una directory contenente più file immagine JPG/JPEG da processare.
  - _Nota: `--image_file` e `--image_dir` sono mutuamente esclusivi e uno dei due è obbligatorio._
- `--prompt_file PERCORSO_FILE_PROMPT` (Opzionale): Specifica il percorso di un file di testo (`.txt`) contenente un prompt personalizzato da utilizzare per la trascrizione. Se omesso, viene utilizzato il file `default_transcription_prompt.txt`.
- `--output_dir PERCORSO_DIRECTORY_OUTPUT` (Opzionale): Specifica la directory dove salvare i file di trascrizione (`.txt`).
  - Se omesso per un file singolo, la trascrizione viene salvata nella stessa directory del file immagine.
  - Se omesso per una directory, le trascrizioni vengono salvate nella directory di input specificata con `--image_dir`.

**Esempi:**

1.  **Trascrivere un singolo file immagine (output nella stessa directory dell'immagine):**

    ```bash
    python -m src.transcription_tool --image_file percorso/alla/mia_immagine.jpg
    ```

2.  **Trascrivere un singolo file immagine con un prompt personalizzato e una directory di output specificata:**

    ```bash
    python -m src.transcription_tool --image_file img/documento.jpeg --prompt_file prompts/mio_prompt.txt --output_dir trascrizioni/
    ```

3.  **Trascrivere tutti i file JPG/JPEG in una directory (output nella stessa directory di input):**

    ```bash
    python -m src.transcription_tool --image_dir foto_da_trascrivere/
    ```

4.  **Trascrivere tutti i file in una directory usando la directory di output `risultati/`:**
    ```bash
    python -m src.transcription_tool --image_dir documenti_scansionati/ --output_dir risultati/
    ```

## Logging

Lo strumento utilizza il modulo `logging` di Python per registrare le sue operazioni:

- **Console:** Messaggi di livello `INFO` e superiori (INFO, WARNING, ERROR, CRITICAL) vengono stampati sulla console, fornendo una panoramica delle operazioni principali, inclusa la classificazione dell'immagine e quale motore di trascrizione (locale o OpenAI) viene utilizzato.
- **File di Log:** Messaggi di livello `DEBUG` e superiori vengono salvati nel file `logs/transcription_tool.log` (creato automaticamente nella root del progetto). Questo file contiene informazioni molto dettagliate utili per il troubleshooting, incluse le predizioni complete del classificatore DiT e le risposte grezze dalle API OpenAI in caso di errori specifici.

## Troubleshooting

- **Errore durante l'inizializzazione dei modelli locali (Hugging Face - DiT o TrOCR):**
  - Assicurarsi che le dipendenze `transformers` e `torch` (o `tensorflow`) siano correttamente installate (vedere `requirements.txt`). Il modello DiT (`microsoft/dit-base-finetuned-rvlcdip`) è necessario per la classificazione. Il modello TrOCR (`microsoft/trocr-base-handwritten`) è attualmente non utilizzato attivamente ma le dipendenze potrebbero essere ancora presenti.
  - Verificare la connessione a internet per il download iniziale dei modelli Hugging Face.
- **Classificazione locale non accurata (con DiT/RVL-CDIP):**
  - Il modello `microsoft/dit-base-finetuned-rvlcdip` classifica le immagini in 16 categorie del dataset RVL-CDIP. La logica in `src/local_image_classifier.py` mappa queste categorie a "handwritten", "typewritten", "other_document_type", o "undetermined" (se la confidenza è bassa o l'etichetta non è mappata esplicitamente).
  - **È fondamentale testare con le proprie immagini** e, se necessario:
    1.  **Affinare il mapping delle etichette:** Modificare le liste di etichette e la logica di mapping in `classify_text_type_local` per adattarle meglio ai risultati del modello sui propri dati.
    2.  **Regolare `MIN_CONFIDENCE_SCORE`:** Questa soglia in `src/local_image_classifier.py` determina la confidenza minima per accettare l'etichetta DiT primaria. Aumentarla può rendere la classificazione più cauta (più "undetermined"), diminuirla la rende più permissiva.
  - Lo script `test_local_classifier.py` è utile per visualizzare tutte le predizioni del modello DiT e aiuta in questo processo di affinamento.
- **Errore `AuthenticationError` (OpenAI API - per la trascrizione):**
  - Verificare che la `OPENAI_API_KEY` nel file `.env` sia corretta e valida.
  - Assicurarsi che il file `.env` sia nella directory principale del progetto.
  - Controllare di avere fondi sufficienti o crediti nel proprio account OpenAI.
- **Errore `NotFoundError` (OpenAI API):**
  - Potrebbe indicare che il modello specificato (es. `gpt-4.1-mini`) non è valido, non è disponibile per il vostro account, o non supporta l'input di immagini. Consultare la documentazione OpenAI per i modelli vision disponibili e, se necessario, modificare la variabile `model_to_use` in `src/llm_handler.py`.
- **Errore `BadRequestError` (OpenAI API):**
  - Questo errore spesso indica un problema con i dati inviati, come un formato immagine non supportato dal modello, un'immagine corrotta, o un prompt malformato. Controllare i log DEBUG per maggiori dettagli, inclusa la risposta dell'API.
- **File di Prompt Non Trovato:**
  - Se si utilizza il prompt di default, assicurarsi che `default_transcription_prompt.txt` esista nella directory principale del progetto.
  - Se si specifica `--prompt_file`, verificare che il percorso sia corretto.
- **Permessi Negati:**
  - Assicurarsi di avere i permessi di lettura per i file immagine e i file di prompt.
  - Assicurarsi di avere i permessi di scrittura per la directory di output.
- **Nessuna Trascrizione Prodotta:**
  - Controllare i log (console e `logs/transcription_tool.log`) per messaggi di errore. La trascrizione potrebbe essere vuota se l'API restituisce un contenuto vuoto o se si verifica un errore durante l'elaborazione.

## Esecuzione dei Test

Il progetto include unit test per verificare la funzionalità dei moduli principali. Per eseguire i test:

1.  Assicurarsi di avere attivato l'ambiente virtuale e installato le dipendenze.
2.  Dalla directory principale del progetto, eseguire:
    ```bash
    python -m unittest discover -s tests -v
    ```
    Questo comando scoprirà ed eseguirà tutti i test nelle directory `tests/`.

## Struttura del Progetto

```
. (Root del Progetto)
├── .env (Da creare, contiene OPENAI_API_KEY - ignorato da Git)
├── .env.example (o dotenv_template.txt - Template per .env)
├── .gitignore
├── default_transcription_prompt.txt (Prompt di default)
├── requirements.txt (Dipendenze Python)
├── README.md (Questo file)
├── logs/
│   └── transcription_tool.log (File di log dettagliato)
├── src/
│   ├── __init__.py
│   ├── transcription_tool.py (Script principale CLI e orchestrazione)
│   ├── file_handler.py (Gestione file: validazione, salvataggio, caricamento prompt)
│   ├── llm_handler.py (Interazione con API OpenAI: configurazione, chiamata LLM)
│   └── local_image_classifier.py (Classificazione locale del tipo di testo con Hugging Face)
└── tests/
    ├── __init__.py
    ├── test_file_handler.py
    ├── test_llm_handler.py
    └── test_transcription_tool.py
```

## Contribuire

(Se applicabile, aggiungere linee guida per la contribuzione).

## Licenza

(Se applicabile, specificare la licenza del progetto).
