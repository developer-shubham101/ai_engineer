"""
Simple local sentiment & tone classifier.

- Uses sentence-transformers for embeddings (all-MiniLM-L6-v2)
- Uses scikit-learn (LogisticRegression) for classification (fast on CPU)
- Auto-trains a tiny default dataset on first run so the module "just works" offline.
- Exposes:
    - predict_sentiment_tone(text) -> {"sentiment":..., "tone":..., "proba": {...}}
    - train_from_examples(examples)
    - save_model(path), load_model(path)
"""

from typing import List, Dict, Tuple
import os
import json
import logging
import threading

import numpy as np

# Core libs (make sure these are in your venv)
# pip install sentence-transformers scikit-learn joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
import joblib
from app.services.utility import (
    get_local_embedding_model_path,
    EMBEDDING_MODEL_NAME,
    BASE_DIR,
    SENTIMENT_ARTIFACTS_DIR,
    get_sentiment_artifact_path,
    normalize_tone_label,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = EMBEDDING_MODEL_NAME  # Use centralized constant
# Where to persist classifier artifacts
DEFAULT_ARTIFACT_DIR = str(SENTIMENT_ARTIFACTS_DIR)
DEFAULT_PIPE_PATH = str(get_sentiment_artifact_path("sentiment_pipeline.joblib"))
DEFAULT_LABELS_PATH = str(get_sentiment_artifact_path("label_encoder.json"))


_lock = threading.RLock()

# Minimal default dataset (tiny). Good for bootstrapping; replace with your data for production.
# Format: (text, sentiment_label, tone_label)
# Note: Tone labels use canonical forms: angry, confused, happy, frustrated, polite, urgent, neutral
_DEFAULT_EXAMPLES = [
    ("Thanks, that helped a lot!", "positive", "polite"),
    ("This is terrible, it keeps failing!", "negative", "angry"),
    ("I can't connect to the VPN, please advise.", "negative", "frustrated"),
    ("Where can I find the holiday policy?", "neutral", "polite"),
    ("Awesome — that solved my issue!", "positive", "happy"),
    ("Why is this broken again?", "negative", "angry"),
    ("Could you please share the process?", "neutral", "polite"),
    ("I need urgent help, my laptop won't boot.", "negative", "urgent"),
    ("Great work on the update!", "positive", "happy"),
    ("I am not sure about the step #3.", "neutral", "confused"),
]


class SentimentToneClassifier:
    def __init__(self,
                 embed_model_name: str = DEFAULT_MODEL_NAME,
                 artifact_dir: str = DEFAULT_ARTIFACT_DIR):
        self.embed_model_name = embed_model_name
        self.artifact_dir = artifact_dir
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Lazy load heavy objects
        self._embedder: SentenceTransformer = None
        self._sent_pipeline: Pipeline = None
        self._sent_label_encoder = None
        # two classifiers: one for sentiment, one for tone
        self._clf_sentiment = None
        self._clf_tone = None

        # try to load existing pipeline
        self._try_load()




    def _get_embedder(self):
        if self._embedder is None:
            logger.info("Loading sentence-transformers model '%s' (CPU)...", self.embed_model_name)
            local_path = get_local_embedding_model_path()
            if local_path.exists():
                self._embedder = SentenceTransformer(str(local_path))
            else:
                self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def _try_load(self):
        """
        Try to load persisted classifier artifacts.
        If artifacts exist, restore classifiers and recreate LabelEncoders from saved class lists.
        If loading fails, bootstrap with the default small dataset (train_from_examples).
        """
        with _lock:
            try:
                # If both pipeline and labels exist, attempt to load them
                if os.path.exists(DEFAULT_PIPE_PATH) and os.path.exists(DEFAULT_LABELS_PATH):
                    logger.info("Loading persisted sentiment pipeline from %s", DEFAULT_PIPE_PATH)
                    obj = joblib.load(DEFAULT_PIPE_PATH)

                    # obj is expected to be a dict containing serialized scikit objects
                    self._clf_sentiment = obj.get("clf_sentiment", None)
                    self._clf_tone = obj.get("clf_tone", None)

                    # load label lists (simple JSON of class lists)
                    le_info = json.load(open(DEFAULT_LABELS_PATH, "r"))
                    self._sent_classes = le_info.get("sentiment_classes", [])
                    self._tone_classes = le_info.get("tone_classes", [])

                    # Recreate LabelEncoder objects (required for inverse_transform)
                    try:
                        from sklearn.preprocessing import LabelEncoder
                        # Sentiment encoder
                        self._sent_label_encoder = LabelEncoder()
                        # LabelEncoder expects a numpy array for .classes_
                        if isinstance(self._sent_classes, (list, tuple)):
                            self._sent_label_encoder.classes_ = np.array(self._sent_classes, dtype=object)
                        else:
                            # If the classes are not in expected format, fallback to classifier classes if available
                            if self._clf_sentiment is not None and hasattr(self._clf_sentiment, "classes_"):
                                self._sent_label_encoder.classes_ = np.array(self._clf_sentiment.classes_, dtype=object)
                            else:
                                raise ValueError("No sentiment classes available to rebuild LabelEncoder")

                        # Tone encoder
                        self._tone_label_encoder = LabelEncoder()
                        if isinstance(self._tone_classes, (list, tuple)):
                            self._tone_label_encoder.classes_ = np.array(self._tone_classes, dtype=object)
                        else:
                            if self._clf_tone is not None and hasattr(self._clf_tone, "classes_"):
                                self._tone_label_encoder.classes_ = np.array(self._clf_tone.classes_, dtype=object)
                            else:
                                raise ValueError("No tone classes available to rebuild LabelEncoder")

                    except Exception as le_ex:
                        # If we cannot recreate encoders, null them and continue.
                        logger.warning("Failed to recreate LabelEncoders from saved data: %s", le_ex)
                        self._sent_label_encoder = None
                        self._tone_label_encoder = None

                    logger.info(
                        "Loaded classifiers. sentiment labels=%s, tone labels=%s, encoders_ok=%s",
                        getattr(self, "_sent_classes", None),
                        getattr(self, "_tone_classes", None),
                        (self._sent_label_encoder is not None and self._tone_label_encoder is not None)
                    )

                else:
                    # No persisted artifacts found — bootstrap small model and persist
                    logger.info("No persisted classifier found — bootstrapping with default examples.")
                    self.train_from_examples(_DEFAULT_EXAMPLES, persist=True)

            except Exception as e:
                # If any loading error occurs, log and bootstrap minimal classifier to ensure functionality
                logger.exception("Failed loading sentiment pipeline: %s. Bootstrapping default classifier.", e)
                try:
                    self.train_from_examples(_DEFAULT_EXAMPLES, persist=True)
                except Exception as t_e:
                    logger.exception("Failed to bootstrap default classifier: %s", t_e)
                    # leave classifiers None — predict() will raise a clear error later

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embedder = self._get_embedder()
        embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embs

    def train_from_examples(self,
                            examples: List[Tuple[str, str, str]],
                            persist: bool = True,
                            C: float = 1.0):
        """
        Train both sentiment and tone classifiers from examples.
        examples: list of (text, sentiment_label, tone_label)
        persist: if True, save artifacts to disk
        """
        with _lock:
            texts = [t for (t, s, tone) in examples]
            sentiments = [s for (t, s, tone) in examples]
            tones = [tone for (t, s, tone) in examples]

            X = self._embed_texts(texts)

            # encode and train sentiment
            sent_le = LabelEncoder()
            y_sent = sent_le.fit_transform(sentiments)
            clf_sent = LogisticRegression(max_iter=1000, C=C, solver="liblinear")
            clf_sent.fit(X, y_sent)

            # tone classifier
            tone_le = LabelEncoder()
            y_tone = tone_le.fit_transform(tones)
            clf_tone = LogisticRegression(max_iter=1000, C=C, solver="liblinear")
            clf_tone.fit(X, y_tone)

            # attach
            self._clf_sentiment = clf_sent
            self._clf_tone = clf_tone
            self._sent_label_encoder = sent_le
            self._tone_label_encoder = tone_le
            self._sent_classes = list(sent_le.classes_)
            self._tone_classes = list(tone_le.classes_)

            logger.info("Trained sentiment classifier (%d classes) and tone classifier (%d classes)",
                        len(self._sent_classes), len(self._tone_classes))

            if persist:
                self._persist()

    def _persist(self):
        """
        Persist classifier artifacts to disk (classifiers + label class lists).
        We don't attempt to persist the sentence-transformer model (heavy) — it's reloaded on demand.
        """
        try:
            out = {
                "clf_sentiment": self._clf_sentiment,
                "clf_tone": self._clf_tone,
                "meta": {
                    "embed_model": self.embed_model_name
                }
            }
            joblib.dump(out, DEFAULT_PIPE_PATH)
            le_info = {
                "sentiment_classes": self._sent_classes,
                "tone_classes": self._tone_classes
            }
            json.dump(le_info, open(DEFAULT_LABELS_PATH, "w"))
            logger.info("Persisted sentiment artifacts to %s", self.artifact_dir)
        except Exception as e:
            logger.exception("Failed to persist sentiment artifacts: %s", e)

    def predict(self, texts: List[str]) -> List[Dict]:
        """
        Predict sentiment & tone for a list of texts.
        Returns a list of dicts:
          {"text": ..., "sentiment": ..., "tone": ..., "proba": {"sentiment": {...}, "tone": {...}}}

        This function now:
        - Never fails; returns defaults (sentiment="unknown", tone="neutral") if classifiers are missing
        - Normalizes tone labels to canonical forms
        - Handles all errors gracefully
        """
        if not texts:
            return []

        with _lock:
            # Ensure the classifiers are present - if not, return defaults
            if self._clf_sentiment is None or self._clf_tone is None:
                logger.warning("Sentiment/tone classifiers not available. Returning defaults.")
                return [
                    {
                        "text": txt,
                        "sentiment": "unknown",
                        "tone": "neutral",
                        "proba": {"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}}
                    }
                    for txt in texts
                ]

            # Ensure LabelEncoders exist; if missing, try to recreate from saved class lists
            # If recreation fails, return defaults instead of raising
            try:
                from sklearn.preprocessing import LabelEncoder
                if self._sent_label_encoder is None:
                    if hasattr(self, "_sent_classes") and self._sent_classes:
                        self._sent_label_encoder = LabelEncoder()
                        self._sent_label_encoder.classes_ = np.array(self._sent_classes, dtype=object)
                    elif hasattr(self._clf_sentiment, "classes_"):
                        self._sent_label_encoder = LabelEncoder()
                        self._sent_label_encoder.classes_ = np.array(self._clf_sentiment.classes_, dtype=object)
                    else:
                        logger.warning("Sentiment LabelEncoder missing. Returning defaults.")
                        return [
                            {
                                "text": txt,
                                "sentiment": "unknown",
                                "tone": "neutral",
                                "proba": {"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}}
                            }
                            for txt in texts
                        ]
                if self._tone_label_encoder is None:
                    if hasattr(self, "_tone_classes") and self._tone_classes:
                        self._tone_label_encoder = LabelEncoder()
                        self._tone_label_encoder.classes_ = np.array(self._tone_classes, dtype=object)
                    elif hasattr(self._clf_tone, "classes_"):
                        self._tone_label_encoder = LabelEncoder()
                        self._tone_label_encoder.classes_ = np.array(self._clf_tone.classes_, dtype=object)
                    else:
                        logger.warning("Tone LabelEncoder missing. Returning defaults.")
                        return [
                            {
                                "text": txt,
                                "sentiment": "unknown",
                                "tone": "neutral",
                                "proba": {"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}}
                            }
                            for txt in texts
                        ]
            except Exception as enc_ex:
                logger.exception("Error ensuring LabelEncoders: %s. Returning defaults.", enc_ex)
                return [
                    {
                        "text": txt,
                        "sentiment": "unknown",
                        "tone": "neutral",
                        "proba": {"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}}
                    }
                    for txt in texts
                ]

            # Embed texts and run classifiers
            X = self._embed_texts(texts)

            # run predictions and probabilities
            sent_preds = self._clf_sentiment.predict(X)
            sent_probas = None
            try:
                sent_probas = self._clf_sentiment.predict_proba(X)
            except Exception:
                # Some classifiers may not implement predict_proba; fallback to one-hot like output
                sent_probas = None

            tone_preds = self._clf_tone.predict(X)
            tone_probas = None
            try:
                tone_probas = self._clf_tone.predict_proba(X)
            except Exception:
                tone_probas = None

            results = []
            for i, txt in enumerate(texts):
                # map integer predictions back to labels using label encoders
                # Use defaults if any step fails
                try:
                    sent_label = self._sent_label_encoder.inverse_transform([sent_preds[i]])[0]
                except Exception as e:
                    logger.warning("Failed to inverse_transform sentiment pred: %s. Using default.", e)
                    sent_label = "unknown"

                try:
                    raw_tone_label = self._tone_label_encoder.inverse_transform([tone_preds[i]])[0]
                    # Normalize tone to canonical form
                    tone_label = normalize_tone_label(raw_tone_label)
                except Exception as e:
                    logger.warning("Failed to inverse_transform tone pred: %s. Using default.", e)
                    tone_label = "neutral"

                # Build probability dicts (if available)
                sent_proba_map = {}
                tone_proba_map = {}
                try:
                    if sent_probas is not None:
                        for j in range(sent_probas.shape[1]):
                            lbl = self._sent_label_encoder.inverse_transform([j])[0]
                            sent_proba_map[lbl] = float(sent_probas[i][j])
                    else:
                        # best-effort one-hot style probability if predict_proba not available
                        for lbl in self._sent_label_encoder.classes_:
                            sent_proba_map[lbl] = 1.0 if lbl == sent_label else 0.0
                except Exception:
                    sent_proba_map = {sent_label: 1.0}

                try:
                    if tone_probas is not None:
                        for j in range(tone_probas.shape[1]):
                            raw_lbl = self._tone_label_encoder.inverse_transform([j])[0]
                            canonical_lbl = normalize_tone_label(raw_lbl)
                            tone_proba_map[canonical_lbl] = float(tone_probas[i][j])
                    else:
                        tone_proba_map = {tone_label: 1.0}
                except Exception:
                    tone_proba_map = {tone_label: 1.0}

                results.append({
                    "text": txt,
                    "sentiment": sent_label,
                    "tone": tone_label,  # Already normalized
                    "proba": {"sentiment": sent_proba_map, "tone": tone_proba_map}
                })

            return results

    def predict_single(self, text: str) -> Dict:
        """
        Predict sentiment & tone for a single text.
        Never fails - always returns a dict with sentiment and tone fields.
        Defaults to sentiment="unknown", tone="neutral" on any error.
        """
        try:
            results = self.predict([text])
            if results:
                return results[0]
        except Exception as e:
            logger.exception("predict_single failed: %s. Returning defaults.", e)
        
        # Fallback defaults
        return {
            "text": text,
            "sentiment": "unknown",
            "tone": "neutral",
            "proba": {"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}}
        }

    def save_pipeline(self, path: str):
        # wrapper to persist to a custom path
        joblib.dump({
            "clf_sentiment": self._clf_sentiment,
            "clf_tone": self._clf_tone,
            "meta": {"embed_model": self.embed_model_name}
        }, path)

    def load_pipeline(self, path: str):
        with _lock:
            obj = joblib.load(path)
            self._clf_sentiment = obj.get("clf_sentiment")
            self._clf_tone = obj.get("clf_tone")
            # attempt to load labels file sibling
            labels_path = os.path.join(os.path.dirname(path), "label_encoder.json")
            if os.path.exists(labels_path):
                info = json.load(open(labels_path, "r"))
                self._sent_classes = info.get("sentiment_classes", [])
                self._tone_classes = info.get("tone_classes", [])
            logger.info("Loaded pipeline from %s", path)

# Convenience singleton for app usage
_global_sentiment = None

def get_global_sentiment():
    global _global_sentiment
    if _global_sentiment is None:
        _global_sentiment = SentimentToneClassifier()
    return _global_sentiment


# cls = SentimentToneClassifier()
# cls.train_from_examples(_DEFAULT_EXAMPLES, persist=True)