# core/self_learning.py
import difflib
import logging
import os
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from .config import Config


class SelfLearningSystem:
    def __init__(self, classifier, aggressive_learning=False):
        self.classifier = classifier
        self.config = Config()
        self.logger = logging.getLogger(__name__)

        self.confidence_threshold = 0.85
        self.buffer_size = 10
        self.aggressive_learning = False

        self.learning_buffer = []
        self.learning_file = "data/self_learning/learning_data.csv"
        self.backup_file = "data/self_learning/learning_data_backup.csv"

        os.makedirs("data/self_learning", exist_ok=True)
        self._initialize_learning_data()

        self.logger.info("Začínam učenie.")

    def _initialize_learning_data(self):
        try:
            learning_data = self.load_learning_data()

            if not learning_data.empty:
                required_columns = ['text', 'category', 'confidence', 'timestamp', 'verified', 'processed']
                missing_columns = [col for col in required_columns if col not in learning_data.columns]

                if missing_columns:
                    self.logger.info(f"Pridávam chýbajúce stĺpce: {missing_columns}")
                    for col in missing_columns:
                        if col == 'verified':
                            learning_data[col] = False
                        elif col == 'processed':
                            learning_data[col] = False
                        elif col == 'timestamp':
                            learning_data[col] = datetime.now().isoformat()
                        else:
                            learning_data[col] = ''

                    learning_data.to_csv(self.learning_file, index=False)
                    self.logger.info("Learning data inicializované s potrebnými stĺpcami")

        except Exception as e:
            self.logger.info("Vytváram nový learning data súbor")
            empty_df = pd.DataFrame(columns=['text', 'category', 'confidence', 'timestamp', 'verified', 'processed'])
            empty_df.to_csv(self.learning_file, index=False)

    def predict_with_learning(self, text: str) -> Tuple[str, Dict[str, float], bool]:
        try:
            category, probabilities = self.classifier.predict(text)
            confidence = max(probabilities.values())

            added_to_learning = False
            if confidence > self.confidence_threshold:
                self._add_to_learning_buffer(text, category, confidence)
                added_to_learning = True
                self.logger.info(f"Pridané do učenia: '{text}' -> {category} ({confidence:.3f})")

            return category, probabilities, added_to_learning

        except Exception as e:
            self.logger.error(f"Chyba v predict_with_learning: {e}")
            return "unknown", {}, False

    def _add_to_learning_buffer(self, text: str, category: str, confidence: float):
        learning_example = {
            'text': text,
            'category': category,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'verified': False,
            'processed': False
        }

        self.learning_buffer.append(learning_example)

        if len(self.learning_buffer) >= 10:
            self._save_learning_data()

        if len(self.learning_buffer) >= self.buffer_size:
            self.logger.info(f"Buffer plný ({len(self.learning_buffer)} príkladov), navrhujem pretrénovanie")

    def _save_learning_data(self):
        try:
            if not self.learning_buffer:
                return

            try:
                existing_df = pd.read_csv(self.learning_file)
            except FileNotFoundError:
                existing_df = pd.DataFrame(
                    columns=['text', 'category', 'confidence', 'timestamp', 'verified', 'processed'])

            new_df = pd.DataFrame(self.learning_buffer)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            combined_df = combined_df.drop_duplicates(subset=['text'])

            combined_df.to_csv(self.learning_file, index=False)
            combined_df.to_csv(self.backup_file, index=False)

            self.logger.info(f"Uložených {len(new_df)} self-learning príkladov")
            self.learning_buffer.clear()

        except Exception as e:
            self.logger.error(f"Chyba pri ukladaní learning dát: {e}")

    def load_learning_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.learning_file)

            required_columns = ['text', 'category', 'confidence', 'timestamp', 'verified', 'processed']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'verified':
                        df[col] = False
                    elif col == 'processed':
                        df[col] = False
                    elif col == 'timestamp':
                        df[col] = datetime.now().isoformat()
                    else:
                        df[col] = ''

            self.logger.info(f"Načítaných {len(df)} self-learning príkladov")
            return df

        except FileNotFoundError:
            self.logger.info("Self-learning súbor neexistuje, vrátený prázdny DataFrame")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Chyba pri načítavaní learning dát: {e}")
            return pd.DataFrame()



    def retrain_with_learning_data(self) -> bool:

        try:
            self.logger.info("Začínam pretrénovanie s self-learning dátami...")

            original_data = pd.read_csv(self.config.DATA_PATH)
            learning_data = self.load_learning_data()

            if learning_data.empty:
                self.logger.info("Žiadne learning dáta pre pretrénovanie")
                return False

            verified_mask = pd.Series([True] * len(learning_data))

            verified_data = learning_data[verified_mask]

            if len(verified_data) == 0:
                self.logger.info("Žiadne overené dáta pre pretrénovanie")
                return False

            self.logger.info(f"Pretrénujem s {len(verified_data)} overenými príkladmi")

            new_data = pd.DataFrame({
                'title': verified_data['text'],
                'category': verified_data['category']
            })

            combined_data = pd.concat([original_data, new_data], ignore_index=True)

            self._backup_current_models()

            self.classifier.feature_extractor.is_fitted = False

            results = self.classifier.train(enable_augmentation=False)

            self.logger.info(f"Pretrénovanie úspešné! Nová presnosť: {results['accuracy']:.3f}")

            self._mark_processed_examples(verified_data)

            return True

        except Exception as e:
            self.logger.error(f"Chyba pri pretrénovaní: {e}")
            self._restore_backup_models()
            return False

    def _backup_current_models(self):
        import shutil
        import glob

        try:
            model_files = glob.glob(os.path.join(self.config.MODELS_DIR, "*"))
            for file_path in model_files:
                if os.path.isfile(file_path):
                    filename = os.path.basename(file_path)
                    shutil.copy2(file_path, f"data/backup_models/{filename}")

            self.logger.info("Aktuálne modely zálohované")
        except Exception as e:
            self.logger.error(f"Chyba pri zálohovaní modelov: {e}")

    def _restore_backup_models(self):
        import shutil
        import glob

        try:
            backup_files = glob.glob("data/backup_models/*")
            for file_path in backup_files:
                if os.path.isfile(file_path):
                    filename = os.path.basename(file_path)
                    shutil.copy2(file_path, os.path.join(self.config.MODELS_DIR, filename))

            self.logger.info("Modely obnovené zo zálohy")
            self.classifier.load_models()
        except Exception as e:
            self.logger.error(f"Chyba pri obnove modelov: {e}")

    def _mark_processed_examples(self, processed_data: pd.DataFrame):
        try:
            learning_data = self.load_learning_data()

            if learning_data.empty:
                return

            if 'processed' not in learning_data.columns:
                learning_data['processed'] = False

            processed_texts = set(processed_data['text'])
            learning_data['processed'] = learning_data['text'].isin(processed_texts)

            learning_data.to_csv(self.learning_file, index=False)
            self.logger.info(f"Označených {len(processed_texts)} príkladov ako spracovaných")

        except Exception as e:
            self.logger.error(f"Chyba pri označovaní príkladov: {e}")

    def get_learning_stats(self) -> Dict[str, any]:
        try:
            learning_data = self.load_learning_data()
            buffer_size = len(self.learning_buffer)

            stats = {
                'buffer_size': buffer_size,
                'saved_examples': len(learning_data),
                'ready_for_retrain': buffer_size >= self.buffer_size,
                'confidence_threshold': self.confidence_threshold
            }

            if not learning_data.empty:
                if 'verified' in learning_data.columns:
                    stats['verified_examples'] = len(learning_data[learning_data['verified'] == True])
                else:
                    stats['verified_examples'] = 0

                if 'confidence' in learning_data.columns:
                    stats['high_confidence_examples'] = len(learning_data[learning_data['confidence'] > 0.85])
                else:
                    stats['high_confidence_examples'] = 0

            if not learning_data.empty and 'category' in learning_data.columns:
                category_counts = learning_data['category'].value_counts().to_dict()
                stats['category_distribution'] = category_counts

            return stats

        except Exception as e:
            self.logger.error(f"Chyba pri získavaní štatistík: {e}")
            return {
                'buffer_size': len(self.learning_buffer),
                'saved_examples': 0,
                'verified_examples': 0,
                'high_confidence_examples': 0,
                'ready_for_retrain': False,
                'confidence_threshold': self.confidence_threshold
            }

    def manual_verification(self, text: str, correct_category: str) -> bool:
        try:
            normalized_category = correct_category.strip().lower()
            if normalized_category not in self.config.CATEGORIES:
                suggestions = difflib.get_close_matches(
                    normalized_category,
                    self.config.CATEGORIES,
                    n=3,
                    cutoff=0.6
                )
                suggestion_text = f" Možné kategórie: {', '.join(suggestions)}." if suggestions else ""
                allowed_categories = ", ".join(self.config.CATEGORIES)
                self.logger.error(
                    "Neplatná kategória pri manuálnom overení: "
                    f"'{correct_category}'. Povolené: {allowed_categories}.{suggestion_text}"
                )
                return False

            learning_example = {
                'text': text,
                'category': normalized_category,
                'confidence': 1.0,
                'timestamp': datetime.now().isoformat(),
                'verified': True,
                'processed': False
            }

            self.learning_buffer.append(learning_example)
            self._save_learning_data()

            self.logger.info(f"Manuálne overené: '{text}' -> {normalized_category}")
            return True

        except Exception as e:
            self.logger.error(f"Chyba pri manuálnom overení: {e}")
            return False
