# tests/test_sentiment.py

import pytest
# Assicurati che la cartella 'app' sia nel percorso di Python
# In una struttura di progetto standard, potresti dover configurare il PYTHONPATH
# GitHub Actions lo gestisce automaticamente se esegui pytest dalla root del progetto.
from app.sentiment_analyzer import analyze_sentiment, LABELS

def test_analyze_sentiment_positive():
    """Verifica che un testo chiaramente positivo venga classificato correttamente."""
    text = "This is a wonderful experience, absolutely fantastic!"
    sentiment, scores = analyze_sentiment(text)
    
    assert sentiment == "positive"
    assert sentiment in LABELS
    assert max(scores, key=scores.get) == "positive"

def test_analyze_sentiment_negative():
    """Verifica che un testo chiaramente negativo venga classificato correttamente."""
    text = "I had a terrible time, it was a disaster."
    sentiment, scores = analyze_sentiment(text)
    
    assert sentiment == "negative"
    assert sentiment in LABELS
    assert max(scores, key=scores.get) == "negative"

def test_analyze_sentiment_neutral():
    """Verifica che un testo neutro venga classificato correttamente."""
    text = "The company will be presenting at the conference next Tuesday."
    sentiment, scores = analyze_sentiment(text)
    
    assert sentiment == "neutral"
    assert sentiment in LABELS
    assert max(scores, key=scores.get) == "neutral"

def test_return_types():
    """Verifica che la funzione restituisca i tipi di dato corretti."""
    text = "This is a test."
    sentiment, scores = analyze_sentiment(text)
    
    assert isinstance(sentiment, str)
    assert isinstance(scores, dict)
    assert all(isinstance(key, str) for key in scores.keys())
    assert all(isinstance(value, float) for value in scores.values())
    assert set(scores.keys()) == set(LABELS)

def test_empty_string():
    """Verifica il comportamento della funzione con una stringa vuota."""
    text = ""
    # Il modello potrebbe restituire 'neutral' o lanciare un errore a seconda dell'implementazione del tokenizer.
    # È buona prassi testare questi casi limite.
    sentiment, _ = analyze_sentiment(text)
    assert sentiment in LABELS

