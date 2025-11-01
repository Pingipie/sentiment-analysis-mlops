# data_generator.py

import requests
import time
import random

API_URL = "http://localhost:8000/predict"

SAMPLE_TEXTS = [
    "This is absolutely amazing, I love it!",
    "The service was terrible from start to finish.",
    "The product is okay, but I expected more features for this price.",
    "Just read an article about the new company policy.",
    "Congratulations on the launch, it looks very promising!",
    "I'm so frustrated, my order is late again.",
    "The webinar will start in 5 minutes.",
    "What a waste of time and money. I want a refund."
]

if __name__ == "__main__":
    print("Avvio del generatore di dati... Invia richieste a", API_URL)
    while True:
        try:
            text_to_send = random.choice(SAMPLE_TEXTS)
            response = requests.post(API_URL, json={"text": text_to_send})
            response.raise_for_status() # Lancia un errore se la richiesta fallisce
            
            print(f"Richiesta inviata. Testo: '{text_to_send[:30]}...'. Risposta: {response.json()['dominant_sentiment']}")
            
        except requests.exceptions.RequestException as e:
            print(f"Errore durante la connessione all'API: {e}")
            
        # Attendi un tempo casuale tra 1 e 5 secondi prima della prossima richiesta
        time.sleep(random.uniform(1, 5))
