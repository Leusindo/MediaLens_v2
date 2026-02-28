# core/model_trainer.py
import logging
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from .config import Config


class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.model = None

    def train_model(self, X_train, y_train, X_test, y_test):
        self.logger.info("Začínam trénovanie modelu...")

        if getattr(self.config, "MODEL_TYPE", "rf") == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 64),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=42,
                verbose=False
            )
            self.logger.info("Trénujem MLP (na embeddings/features)...")
        else:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.7,
                class_weight="balanced",
                random_state=42,
                bootstrap=True
            )
            self.logger.info("Trénujem Random Forest...")

        self.model.fit(X_train, y_train)

        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)

        self.logger.info(f"Trénovacia presnosť: {train_accuracy:.4f}")
        self.logger.info(f"Testovacia presnosť: {test_accuracy:.4f}")

        y_pred = self.model.predict(X_test)
        self.logger.info("\nClassification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")

        return self.model

    def save_model(self):
        if self.model is not None:
            os.makedirs(self.config.MODELS_DIR, exist_ok=True)
            model_path = os.path.join(self.config.MODELS_DIR, "trained_model.joblib")
            joblib.dump(self.model, model_path)
            self.logger.info(f"Model uložený do: {model_path}")

    def load_model(self):
        model_path = os.path.join(self.config.MODELS_DIR, "trained_model.joblib")
        self.model = joblib.load(model_path)
        self.logger.info("Model načítaný")
        return self.model
