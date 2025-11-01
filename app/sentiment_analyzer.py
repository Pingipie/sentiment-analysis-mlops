# app/sentiment_analyzer.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# --- Caricamento del Modello ---
# Il modello e il tokenizer vengono caricati una sola volta quando il modulo viene importato,
# ottimizzando le performance per chiamate multiple.
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    # Sposta il modello sulla GPU se disponibile, altrimenti usa la CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
except Exception as e:
    # Gestisce il caso in cui il modello non possa essere caricato (es. offline)
    print(f"Errore durante il caricamento del modello: {e}")
    tokenizer = None
    model = None

# Mappatura delle etichette come definite dal modello su Hugging Face
LABELS = ['negative', 'neutral', 'positive']

def analyze_sentiment(text: str) -> (str, dict):
    """
    Analizza il sentiment di una stringa di testo.

    Args:
        text (str): Il testo da analizzare.

    Returns:
        tuple: Una tupla contenente:
               - str: Il sentiment dominante ('negative', 'neutral', 'positive').
               - dict: Un dizionario con i punteggi di confidenza per ciascun sentiment.
    
    Raises:
        RuntimeError: Se il modello non è stato caricato correttamente.
    """
    if not model or not tokenizer:
        raise RuntimeError("Modello di sentiment analysis non inizializzato. Controlla la connessione internet o il path del modello.")

    # Tokenizza l'input e lo sposta sul dispositivo corretto (CPU/GPU)
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    
    # Esegue l'inferenza del modello senza calcolare i gradienti
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Estrae i punteggi grezzi (logits) e li sposta sulla CPU per l'elaborazione con NumPy/SciPy
    scores = output[0][0].cpu().numpy()
    
    # Applica la funzione softmax per convertire i punteggi in probabilità
    probabilities = softmax(scores)
    
    # Crea un dizionario con i punteggi di confidenza per ogni etichetta
    confidence_scores = {label: float(prob) for label, prob in zip(LABELS, probabilities)}
    
    # Determina il sentiment dominante
    dominant_sentiment = LABELS[probabilities.argmax()]
    
    return dominant_sentiment, confidence_scores

