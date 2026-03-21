# core/classifier.py
import logging
import os
import re
from typing import List, Dict, Tuple, Any, Optional
import joblib
import numpy as np
from tqdm import tqdm
from .config import Config
from .data_processor import DataProcessor
from .feature_extractor import HybridFeatureExtractor
from .model_trainer import ModelTrainer


class NewsClassifier:
    MODEL_FEATURE_PRESETS = {
        'rf': {
            'use_bert': True,
            'use_tfidf': True,
            'use_sentence_embeddings': False,
        },
        'mlp': {
            'use_bert': False,
            'use_tfidf': False,
            'use_sentence_embeddings': True,
        }
    }

    def __init__(self, use_bert: bool = False, use_tfidf: bool = True, use_sentence_embeddings: bool = False, model_type: Optional[str] = None):

        self.config = Config()
        self.logger = logging.getLogger(__name__)

        if model_type in self.MODEL_FEATURE_PRESETS:
            preset = self.MODEL_FEATURE_PRESETS[model_type]
            use_bert = preset['use_bert']
            use_tfidf = preset['use_tfidf']
            use_sentence_embeddings = preset['use_sentence_embeddings']

        self.config.USE_BERT = use_bert
        self.config.USE_TFIDF = use_tfidf
        self.config.USE_SENTENCE_EMBEDDINGS = use_sentence_embeddings
        Config.USE_BERT = use_bert
        Config.USE_TFIDF = use_tfidf
        Config.USE_SENTENCE_EMBEDDINGS = use_sentence_embeddings
        if model_type is not None:
            self.config.MODEL_TYPE = model_type
            Config.MODEL_TYPE = model_type

        self.feature_extractor = HybridFeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.data_processor = DataProcessor()

        self.is_trained = False
        self.is_loaded = False
        self.ensemble_models: Dict[str, Dict[str, Any]] = {}
        self.last_winning_model: Optional[str] = None

        self.logger.info(
            f"Classifier inicializovaný - BERT: {self.config.USE_BERT}, TF-IDF: {self.config.USE_TFIDF}, "
            f"SENT: {self.config.USE_SENTENCE_EMBEDDINGS}, MODEL: {getattr(self.config, 'MODEL_TYPE', 'rf')}"
        )
        self.logger.info(f"Používané zariadenie: {self.config.DEVICE}")

    def train(self, enable_augmentation: bool = True) -> Dict[str, Any]:

        self.logger.info("=== ZAČÍNAM TRÉNOVANIE KLASIFIKÁTORA ===")

        try:
            df = self.data_processor.load_data()
            self.logger.info(f"Načítaných {len(df)} trénovacích príkladov")

            if enable_augmentation:
                df = self.data_processor.augment_data(df)
                self.logger.info(f"Po augmentácii: {len(df)} príkladov")

            X, y, original_labels = self.data_processor.preprocess_data(df)

            X_train, X_test, y_train, y_test = self.data_processor.split_data(X, y)

            self.logger.info("Začínam feature extraction...")
            self.feature_extractor.fit(X_train)
            X_train_features = self.feature_extractor.transform(X_train)
            X_test_features = self.feature_extractor.transform(X_test)

            self.logger.info(f"Trénovacie features: {X_train_features.shape}")
            self.logger.info(f"Testovacie features: {X_test_features.shape}")

            trained_model = self.model_trainer.train_model(
                X_train_features, y_train, X_test_features, y_test
            )

            self._save_models()

            results = self._evaluate_training(X_test_features, y_test, original_labels)

            self.is_trained = True
            self.is_loaded = True

            self.logger.info("=== TRÉNOVANIE ÚSPEŠNE DOKONČENÉ ===")
            return results

        except Exception as e:
            self.logger.error(f"Chyba pri trénovaní: {e}")
            raise

    def _save_models(self):
        self.logger.info("Ukladám modely...")

        os.makedirs(self.config.MODELS_DIR, exist_ok=True)

        self.feature_extractor.save()

        self.model_trainer.save_model()

        self.data_processor.save_label_encoder()

        model_type = getattr(self.config, 'MODEL_TYPE', 'rf')
        config_payload = {
            'use_bert': self.config.USE_BERT,
            'use_tfidf': self.config.USE_TFIDF,
            'use_sentence_embeddings': self.config.USE_SENTENCE_EMBEDDINGS,
            'model_type': model_type,
            'categories': self.config.CATEGORIES,
            'feature_dimensions': {
                'bert_original': self.config.BERT_EMBEDDING_DIM,
                'bert_reduced': self.config.REDUCED_BERT_DIM,
                'tfidf': self.config.TFIDF_MAX_FEATURES
            }
        }

        config_path = os.path.join(self.config.MODELS_DIR, 'training_config.joblib')
        typed_config_path = os.path.join(self.config.MODELS_DIR, f'training_config_{model_type}.joblib')
        joblib.dump(config_payload, config_path)
        joblib.dump(config_payload, typed_config_path)

        self.logger.info(f"Všetky modely uložené do: {self.config.MODELS_DIR}")

    def _evaluate_training(self, X_test, y_test, original_labels) -> Dict[str, Any]:
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        y_pred = self.model_trainer.model.predict(X_test)
        y_pred_proba = self.model_trainer.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        confidence_scores = np.max(y_pred_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)

        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'average_confidence': avg_confidence,
            'test_set_size': len(X_test),
            'feature_dimensions': X_test.shape[1],
            'categories': self.data_processor.label_encoder.classes_.tolist()
        }

        self.logger.info(f"Konečná presnosť: {accuracy:.4f}")
        self.logger.info(f"Priemerná istota: {avg_confidence:.4f}")
        self.logger.info(f"Počet features: {X_test.shape[1]}")

        return results

    def _apply_model_preset(self, model_type: str):
        preset = self.MODEL_FEATURE_PRESETS[model_type]
        self.config.USE_BERT = preset['use_bert']
        self.config.USE_TFIDF = preset['use_tfidf']
        self.config.USE_SENTENCE_EMBEDDINGS = preset['use_sentence_embeddings']
        self.config.MODEL_TYPE = model_type
        Config.USE_BERT = preset['use_bert']
        Config.USE_TFIDF = preset['use_tfidf']
        Config.USE_SENTENCE_EMBEDDINGS = preset['use_sentence_embeddings']
        Config.MODEL_TYPE = model_type

    def _load_single_model(self, model_type: str) -> bool:
        if model_type not in self.MODEL_FEATURE_PRESETS:
            self.logger.error(f"Neplatný typ modelu: {model_type}")
            return False

        self._apply_model_preset(model_type)

        model_path = os.path.join(self.config.MODELS_DIR, f"trained_model_{model_type}.joblib")
        legacy_model_path = os.path.join(self.config.MODELS_DIR, "trained_model.joblib")
        if not os.path.exists(model_path) and not os.path.exists(legacy_model_path):
            self.logger.warning(f"Preskakujem {model_type}: model súbor neexistuje")
            return False

        feature_extractor = HybridFeatureExtractor()
        model_trainer = ModelTrainer()
        data_processor = DataProcessor()

        feature_extractor.load()
        model_trainer.load_model(model_type=model_type)
        data_processor.load_label_encoder()

        self.ensemble_models[model_type] = {
            'feature_extractor': feature_extractor,
            'model_trainer': model_trainer,
            'data_processor': data_processor,
        }
        return True

    def load_models(self, model_type: Optional[str] = None) -> bool:
        try:
            self.logger.info("Načítavam natrénované modely...")

            label_path = os.path.join(self.config.MODELS_DIR, 'label_encoder.joblib')
            if not os.path.exists(label_path):
                self.logger.error("Chýbajúci súbor: label_encoder.joblib")
                return False

            self.ensemble_models = {}
            targets = [model_type] if model_type else ['rf', 'mlp']
            loaded_types = [m for m in targets if self._load_single_model(m)]

            if not loaded_types:
                self.logger.error("Nepodarilo sa načítať žiadny model (rf/mlp)")
                return False

            primary_model_type = loaded_types[0]
            self.feature_extractor = self.ensemble_models[primary_model_type]['feature_extractor']
            self.model_trainer = self.ensemble_models[primary_model_type]['model_trainer']
            self.data_processor = self.ensemble_models[primary_model_type]['data_processor']
            self.config.MODEL_TYPE = primary_model_type
            Config.MODEL_TYPE = primary_model_type

            self.is_loaded = True
            self.logger.info("Všetky modely úspešne načítané")
            self.logger.info(f"Načítané modely: {', '.join(loaded_types)}")

            return True

        except Exception as e:
            self.logger.error(f"Chyba pri načítavaní modelov: {e}")
            return False

    def predict(self, text: str, return_confidence: bool = True) -> Tuple[str, Dict[str, float]]:

        if not self.is_loaded and not self.load_models():
            raise ValueError("Modely nie sú načítané a nepodarilo sa ich načítať!")

        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            raise ValueError("Text je prázdny po vyčistení!")

        if not self.ensemble_models:
            self.ensemble_models = {
                self.config.MODEL_TYPE: {
                    'feature_extractor': self.feature_extractor,
                    'model_trainer': self.model_trainer,
                    'data_processor': self.data_processor,
                }
            }

        candidates = []
        for active_model_type, bundle in self.ensemble_models.items():
            try:
                model_features = bundle['feature_extractor'].transform([cleaned_text])
                prediction = bundle['model_trainer'].model.predict(model_features)[0]
                probabilities = bundle['model_trainer'].model.predict_proba(model_features)[0]
                predicted_label = bundle['data_processor'].label_encoder.inverse_transform([prediction])[0]
                prob_dict = {
                    category: float(prob) for category, prob in zip(
                        bundle['data_processor'].label_encoder.classes_,
                        probabilities
                    )
                }
                confidence = float(max(probabilities))
                candidates.append((confidence, predicted_label, prob_dict, active_model_type))
            except Exception as exc:
                self.logger.warning("Model %s zlyhal pri predikcii: %s", active_model_type, exc)

        if not candidates:
            raise ValueError("Žiadny model nedokázal spraviť predikciu")

        confidence, predicted_label, prob_dict, winning_model = max(candidates, key=lambda x: x[0])
        self.last_winning_model = winning_model
        self.logger.debug(f"Vybraný model: {winning_model} (confidence={confidence:.3f})")

        return predicted_label, prob_dict

    def classify(self, text: str) -> Dict[str, Any]:

        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            raise ValueError("Text nemôže byť prázdny.")

        try:
            label, probabilities = self.predict(cleaned_text)
        except Exception as exc:
            self.logger.warning("Modelové predikovanie zlyhalo, používam fallback: %s", exc)
            label, probabilities = self._fallback_prediction(text)

        confidence = float(max(probabilities.values()) if probabilities else 0.0)

        return {
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "cleaned_text": cleaned_text,
            "model_used": self.last_winning_model or getattr(self.config, "MODEL_TYPE", None),
        }

    def predict_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:

        if not self.is_loaded and not self.load_models():
            raise ValueError("Modely nie sú načítané!")

        if not texts:
            return []

        cleaned_texts = [self._clean_text(text) for text in texts]
        valid_texts = [text for text in cleaned_texts if text]

        if not valid_texts:
            raise ValueError("Žiadny validný text po čistení!")

        if show_progress:
            self.logger.info(f"Spracovávam {len(valid_texts)} textov...")

        features = self.feature_extractor.transform(valid_texts)

        predictions = self.model_trainer.model.predict(features)
        probabilities = self.model_trainer.model.predict_proba(features)

        predicted_labels = self.data_processor.label_encoder.inverse_transform(predictions)

        results = []
        iterator = zip(texts, predicted_labels, probabilities)

        if show_progress:
            iterator = tqdm(iterator, total=len(texts), desc="Klasifikácia")

        for original_text, label, probs in iterator:
            prob_dict = {
                category: float(prob) for category, prob in zip(
                    self.data_processor.label_encoder.classes_,
                    probs
                )
            }

            results.append({
                'original_text': original_text,
                'cleaned_text': self._clean_text(original_text),
                'predicted_category': label,
                'probabilities': prob_dict,
                'confidence': float(max(probs)),
                'is_high_confidence': max(probs) > 0.7
            })

        return results

    def _clean_text(self, text: str) -> str:

        if not isinstance(text, str):
            return ""

        text = re.sub(r'\s+', ' ', text.strip())

        text = re.sub(r'[^\w\sáäčďéíľĺňóôřŕšťúýžÁÄČĎÉÍĽĹŇÓÔŘŔŠŤÚÝŽ\-\'",.!?;]', '', text)

        return text.lower()

    def _fallback_prediction(self, text: str) -> Tuple[str, Dict[str, float]]:
        self.logger.warning(f"Používam fallback klasifikáciu pre: {text}")

        text_lower = text.lower()

        keywords = {
            'clickbait': ['šokujúce', 'neuveríte', 'kliknite', 'zistíte', 'tajomstvo', 'odhalenie'],
            'false_news': ['zákaz', 'ruší', 'zatvorené', 'zakazuje', 'zakázal', 'povinné'],
            'conspiracy': ['tajné', 'elity', 'ovládajú', 'alien', 'mimozemšťan', 'bunkor'],
            'propaganda': ['jediná', 'záchrana', 'naša strana', 'lídra', 'úspechy'],
            'satire': ['zrušený', 'zakázal dážď', 'povinné nosenie', 'bryndzové halušky'],
            'misleading': ['všetkých', 'zázračne', 'úplne', 'absolútne', '100%'],
            'biased': ['neschopný', 'zlyhal', 'podvodník', 'hlúpy', 'kritizuje'],
            'legitimate': ['schválila', 'oznámil', 'vydalo', 'objavili', 'otvorí']
        }

        scores = {category: 0 for category in self.config.CATEGORIES}

        for category, words in keywords.items():
            for word in words:
                if word in text_lower:
                    scores[category] += 1

        max_score = max(scores.values())
        if max_score > 0:
            predicted_category = max(scores, key=scores.get)
        else:
            predicted_category = 'legitimate'

        total_score = sum(scores.values()) or 1
        prob_dict = {
            category: score / total_score for category, score in scores.items()
        }

        return predicted_category, prob_dict

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"status": "Modely nie sú načítané"}

        info = {
            "status": "Natrénovaný a načítaný",
            "feature_extractor": {
                "uses_bert": self.config.USE_BERT,
                "uses_tfidf": self.config.USE_TFIDF,
                "bert_model": self.config.BERT_MODEL_NAME
            },
            "classifier": {
                "type": "RandomForest",
                "n_estimators": self.config.N_ESTIMATORS
            },
            "categories": self.config.CATEGORIES,
            "device": self.config.DEVICE
        }

        if hasattr(self.model_trainer, 'model') and self.model_trainer.model is not None:
            info["classifier"]["n_features"] = (
                self.model_trainer.model.n_features_in_
                if hasattr(self.model_trainer.model, 'n_features_in_')
                else "Unknown"
            )

        return info

    def evaluate_custom_data(self, texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        from sklearn.metrics import classification_report, accuracy_score

        if not self.is_loaded:
            raise ValueError("Modely musia byť načítané!")

        if len(texts) != len(true_labels):
            raise ValueError("Texty a labels musia mať rovnakú dĺžku!")

        results = self.predict_batch(texts, show_progress=True)
        predicted_labels = [result['predicted_category'] for result in results]

        accuracy = accuracy_score(true_labels, predicted_labels)
        report = classification_report(true_labels, predicted_labels, output_dict=True)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels,
                              labels=self.data_processor.label_encoder.classes_)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'sample_size': len(texts),
            'predictions': results
        }


def create_classifier(use_bert: bool = True, use_tfidf: bool = True) -> NewsClassifier:
    return NewsClassifier(use_bert=use_bert, use_tfidf=use_tfidf)


def test_classifier():
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testovanie NewsClassifier...")

    classifier = create_classifier()

    info = classifier.get_model_info()
    print(f"Model info: {info}")

    test_texts = [
        "Šokujúce odhalenie v Bratislave!",
        "Vláda schválila nový rozpočet",
        "Tajné spolky ovládajú parlament",
        "Nový zákon zakazuje bicykle"
    ]

    print("\nTestovacie predikcie:")
    for text in test_texts:
        try:
            category, probs = classifier.predict(text)
            print(f"  '{text}' -> {category} (istota: {max(probs.values()):.3f})")
        except Exception as e:
            print(f"  Chyba pre '{text}': {e}")


if __name__ == "__main__":
    test_classifier()
