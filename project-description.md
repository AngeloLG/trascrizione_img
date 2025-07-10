### Architettura e Flusso Generale

L'applicazione è uno strumento a riga di comando (CLI) scritto in Python, progettato per orchestrare un processo di **trascrizione di testo da immagini**. Il codice è ben strutturato e suddiviso in moduli, ciascuno con una responsabilità specifica, il che è un'ottima pratica di ingegneria del software.

I moduli principali sono:

- `transcription_tool.py`: È il "cervello" dell'applicazione. Gestisce gli input dell'utente, orchestra le operazioni e decide quale strumento utilizzare per la trascrizione.
- `file_handler.py`: Si occupa di tutte le operazioni sui file: leggere le immagini, validarle, caricare il testo del "prompt" e salvare le trascrizioni finali.
- `local_image_classifier.py`: Contiene la logica per analizzare un'immagine con un modello di classificazione locale (scaricato da Hugging Face) per determinare il tipo di documento (es. manoscritto, dattiloscritto).
- `llm_handler.py`: Gestisce tutta la comunicazione con i modelli linguistici (LLM), sia quelli locali (attualmente inattivi) sia l'API di OpenAI.

### Operazioni Passo-Passo

Quando esegui lo script, ecco cosa succede:

**1. Avvio e Lettura degli Argomenti**

- Lo script parte dalla funzione `main()` in `transcription_tool.py`.
- Per prima cosa, analizza gli argomenti che hai fornito sulla riga di comando. Devi specificare obbligatoriamente o un singolo file immagine (`--image_file`) o una cartella di immagini (`--image_dir`).
- Puoi anche fornire argomenti opzionali, come un file di "prompt" personalizzato (`--prompt_file`) o una cartella di output per le trascrizioni (`--output_dir`).

**2. Caricamento del Prompt**

- Il "prompt" è l'insieme di istruzioni che viene inviato al modello AI per guidarlo nella trascrizione.
- Se hai specificato un file con `--prompt_file`, lo script carica il contenuto di quel file.
- Altrimenti, carica il prompt di default dal file `default_transcription_prompt.txt` che si trova nella cartella principale del progetto. Questo prompt è ottimizzato per la trascrizione di documenti italiani dattiloscritti del '900.

**3. Elaborazione delle Immagini**

A questo punto, il flusso si biforca a seconda che tu abbia specificato un file singolo o una directory:

- **Caso Directory (`--image_dir`):** Lo script esamina tutti i file nella cartella specificata. Se trova file con estensione supportata (`.jpg`, `.jpeg`), li processa uno alla volta.
- **Caso File Singolo (`--image_file`):** Viene processato solo il file immagine specificato.

**4. Processo di Trascrizione per una Singola Immagine (la fase cruciale)**

Questa è la parte più importante. Ogni immagine segue questi passaggi:

- **a. Validazione del File:** Lo script si assicura che il file esista e sia un'immagine valida.

- **b. Classificazione Locale dell'Immagine:**

  - **Scopo:** Per ottimizzare i costi e le prestazioni, lo script **prima classifica l'immagine localmente**, senza usare l'API a pagamento di OpenAI.
  - **Come:** Utilizza il modello `microsoft/dit-base-finetuned-rvlcdip` da Hugging Face. Questo modello è addestrato per riconoscere tipi di documenti (lettere, report, fatture, ecc.). Il codice mappa l'output di questo modello in categorie più semplici come `"handwritten"` (manoscritto), `"typewritten"` (dattiloscritto) o `"other_document_type"`.
  - **Logica:** Questa classificazione è fondamentale perché determina il passo successivo. L'esito viene stampato a console.

- **c. Scelta del Motore di Trascrizione (Decisione Condizionale):**

  - **Se l'immagine è classificata come `"handwritten"`:** Il `README.md` e il codice indicano che l'intenzione originale era di usare un modello OCR locale (`microsoft/trocr-base-handwritten`) per i manoscritti. Tuttavia, il `README` chiarisce che **questa funzionalità è attualmente sospesa**. Di conseguenza, anche se il codice ha ancora la logica per tentare questa strada, allo stato attuale il flusso per i manoscritti si interromperebbe o fallirebbe (come da commenti nel codice), a meno che non venga inviato comunque a OpenAI. _Nella versione attuale del codice che ho analizzato, la trascrizione con modello OpenAI è il fallback per tutti i tipi di documento che non siano manoscritti. I documenti `handwritten` attivano una logica separata che al momento risulta in un vicolo cieco o in un errore._
  - **Se l'immagine è classificata in qualsiasi altro modo (`"typewritten"`, ecc.):** Lo script procede utilizzando l'API di OpenAI.

- **d. Trascrizione tramite OpenAI LLM:**

  - L'immagine viene convertita in un formato (Base64) che può essere inviato all'API di OpenAI.
  - Viene costruita una richiesta che include sia il **prompt** (caricato al punto 2) sia **l'immagine**.
  - Questa richiesta viene inviata al modello OpenAI specificato nel codice, ovvero `gpt-4.1-mini`.
  - Lo script attende la risposta dall'API, che contiene la trascrizione del testo.

- **e. Salvataggio della Trascrizione:**
  - Una volta ottenuta la trascrizione, questa viene salvata in un file di testo (`.txt`).
  - Il nome del file di testo è identico a quello dell'immagine originale (es. `mia_immagine.jpg` -> `mia_immagine.txt`).
  - La cartella di salvataggio è quella specificata con `--output_dir` o, in assenza, la stessa cartella dell'immagine di input.

**5. Logging**

- Durante tutte queste operazioni, lo script registra informazioni utili sia sulla **console** (per darti un feedback immediato) sia in un **file di log** (`logs/transcription_tool.log`).
- Il file di log è molto più dettagliato e contiene informazioni di debug che possono essere preziose per capire cosa è andato storto in caso di problemi.

In sintesi, questo software è un **orchestratore intelligente** che prima cerca di capire la natura di un'immagine usando un modello locale gratuito e poi, sulla base di questa analisi, decide se usare un potente (ma a pagamento) modello cloud per eseguire la trascrizione vera e propria, gestendo tutti i file e i dati in modo pulito e robusto.
