# app/web_app.py

from flask import Flask, request, jsonify
from prometheus_client import Counter, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from app.sentiment_analyzer import analyze_sentiment

app = Flask(__name__)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

predictions_total = Counter(
    'sentiment_predictions_total', 
    'Total number of sentiment predictions', 
    ['sentiment']
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Il campo 'text' è richiesto"}), 400

        text = data['text']
        sentiment, scores = analyze_sentiment(text)
        
        predictions_total.labels(sentiment).inc()

        return jsonify({
            "text": text,
            "dominant_sentiment": sentiment,
            "confidence_scores": scores
        })
    except Exception as e:
        return jsonify({"error": "Errore interno del server"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
