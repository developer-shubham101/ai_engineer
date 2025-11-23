from app.services.sentiment_classifier import get_global_sentiment

c = get_global_sentiment()
examples = [
    "I am furious that it keeps crashing!",
    "Thanks a lot, worked like a charm",
    "Can someone help me fix this error?",
]
for ex in examples:
    print(ex, "->", c.predict_single(ex))
