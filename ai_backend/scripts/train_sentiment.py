from app.services.sentiment_classifier import SentimentToneClassifier

cls = SentimentToneClassifier()
# examples: list of tuples (text, sentiment_label, tone_label)
my_examples = [
    ("My laptop crashed and I lost work", "negative", "angry"),
    ("Thank you for the help", "positive", "polite"),
    ("Thanks, that helped a lot!", "positive", "polite"),
    ("This is terrible, it keeps failing!", "negative", "angry"),
    ("I can't connect to the VPN, please advise.", "negative", "frustrated"),
    ("Where can I find the holiday policy?", "neutral", "curious"),
    ("Awesome — that solved my issue!", "positive", "happy"),
    ("Why is this broken again?", "negative", "angry"),
    ("Could you please share the process?", "neutral", "polite"),
    ("I need urgent help, my laptop won't boot.", "negative", "urgent"),
    ("Great work on the update!", "positive", "appreciative"),
    ("I am not sure about the step #3.", "neutral", "confused"),
    #...
]
cls.train_from_examples(my_examples, persist=True)
