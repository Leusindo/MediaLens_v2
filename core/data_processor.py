# core/data_processor.py
import logging
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import Config
import random
import numpy as np

random.seed(42)
np.random.seed(42)


class DataProcessor:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def load_data(self) -> pd.DataFrame:
        self.logger.info(f"Nacitavam data z: {self.config.DATA_PATH}")

        try:
            df = pd.read_csv(self.config.DATA_PATH)
            self.logger.info(f"Data uspesne nacitane. Tvar: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Chyba pri nacitavani dat: {e}")
            raise

    def augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Zacinam augmentaciu dat...")

        augmented_rows = []

        variations = {
            'šokujúce': ['šokované', 'šokovaní', 'šokujúci', 'šokujúca', 'šok'],
            'neuveríte': ['neuveriteľné', 'neuveriteľný', 'neveríte', 'neverý'],
            'kliknite': ['kliknite', 'stlačte', 'tlačte'],
            'zistíte': ['zistite', 'zisťujte', 'zisťuje'],
            'zakazuje': ['zakázal', 'zakáže', 'zakázalo', 'zákaz', 'zakázaný'],
            'ruší': ['zrušil', 'zruší', 'zrušia', 'zrušenie', 'rušenie'],
            'zákaz': ['zakázanie', 'zákazy', 'zakázaný'],
            'tajné': ['tajomstvo', 'tajný', 'tajná', 'utajené', 'skryté'],
            'elity': ['elita', 'elitám', 'elitami'],
            'ovládajú': ['ovládajú', 'ovláda', 'riadia', 'kontrolujú'],
            'jediná': ['jediný', 'jediné', 'jedinečná', 'jedinečný'],
            'záchrana': ['záchranu', 'záchrany', 'zachráni', 'zachraňuje'],
            'zrušený': ['zrušená', 'zrušené', 'zrušil', 'zruší'],
            'povinné': ['povinný', 'povinná', 'povinne', 'povinnosť'],
            'nový': ['nová', 'nové', 'noví', 'nového', 'novú'],
            'vláda': ['vládny', 'vládne', 'vládou', 'vlády'],
            'ministerstvo': ['ministerstvom', 'ministerstve', 'ministerstvá'],
        }

        for _, row in df.iterrows():
            augmented_rows.append(row)

            title = row['title'].lower()
            category = row['category']

            variations_created = 0
            max_variations = 2

            for base_word, variants in variations.items():
                if base_word in title and variations_created < max_variations:
                    for variant in random.sample(variants, min(2, len(variants))):
                        if random.random() > 0.6:
                            new_title = title.replace(base_word, variant)

                            if new_title != title and len(new_title) > len(title) * 0.8:
                                new_row = row.copy()
                                new_row['title'] = new_title
                                augmented_rows.append(new_row)
                                variations_created += 1

                                if variations_created >= max_variations:
                                    break
                    if variations_created >= max_variations:
                        break

            if variations_created < max_variations:
                variations_list = [
                    title.capitalize(),
                    title.upper(),
                    title + "!",
                    title.replace("!", "?"),
                ]

                for variation in random.sample(variations_list, min(1, len(variations_list))):
                    if variation != title:
                        new_row = row.copy()
                        new_row['title'] = variation
                        augmented_rows.append(new_row)
                        variations_created += 1

                        if variations_created >= max_variations:
                            break

        result_df = pd.DataFrame(augmented_rows).drop_duplicates(subset=['title'])
        self.logger.info(f"Augmentacia dokoncena: {len(df)} → {len(result_df)} prikladov")

        self.logger.info("Rozdelenie kategorii po augmentacii:")
        for category, count in result_df['category'].value_counts().items():
            self.logger.info(f"  {category}: {count} prikladov")

        return result_df

    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        self.logger.info("Spustam preprocessing dat...")

        df = df.dropna(subset=['title', 'category'])

        df['cleaned_title'] = df['title'].apply(self._clean_text)

        df = df[df['cleaned_title'].str.len() > 0]

        y_encoded = self.label_encoder.fit_transform(df['category'])
        self.is_fitted = True

        self.logger.info(f"Preprocessing dokoncený. Pocet prikladov: {len(df)}")
        self.logger.info(f"Rozdelenie kategorii:\n{df['category'].value_counts()}")

        return df['cleaned_title'].tolist(), y_encoded, df['category'].tolist()

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = re.sub(r'\s+', ' ', text.strip())

        text = re.sub(r'[^\w\sáäčďéíľĺňóôřŕšťúýžÁÄČĎÉÍĽĹŇÓÔŘŔŠŤÚÝŽ\-\'",.!?;]', '', text)

        return text.lower()

    def split_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )

        self.logger.info(f"Trénovacie dáta: {len(X_train)}")
        self.logger.info(f"Testovacie dáta: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def save_label_encoder(self):
        if self.is_fitted:
            os.makedirs(self.config.MODELS_DIR, exist_ok=True)
            joblib.dump(self.label_encoder,
                        os.path.join(self.config.MODELS_DIR, 'label_encoder.joblib'))
            self.logger.info("Label encoder ulozeny")

    def load_label_encoder(self):
        path = os.path.join(self.config.MODELS_DIR, 'label_encoder.joblib')
        self.label_encoder = joblib.load(path)
        self.is_fitted = True
        self.logger.info("Label encoder nacitany")