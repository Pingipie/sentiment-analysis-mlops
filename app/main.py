# app/main.py

from sentiment_analyzer import analyze_sentiment

def run_examples():
    """
    Esegue alcuni esempi di analisi del sentiment e stampa i risultati.
    """
    print("--- Esecuzione Esempi di Analisi del Sentiment ---\n")
    
    test_texts = [
        "I love the new features in this app! It's incredibly useful and well-designed.",
        "The company announced its quarterly earnings report this morning.",
        "I'm very disappointed with the customer service. My issue has not been resolved for weeks."
    ]
    
    for text in test_texts:
        try:
            sentiment, scores = analyze_sentiment(text)
            print(f"Testo: '{text}'")
            print(f"  -> Sentiment Rilevato: {sentiment}")
            print(f"  -> Punteggi di Confidenza: {scores}\n")
        except RuntimeError as e:
            print(f"Errore durante l'analisi del testo: '{text}'. Dettagli: {e}")
            break

if __name__ == "__main__":
    run_examples()

