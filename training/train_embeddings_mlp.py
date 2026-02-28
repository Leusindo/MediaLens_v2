# training/train_embeddings_mlp.py
import logging
import os
import sys
from core.classifier import NewsClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training_embeddings_mlp.log"),
            logging.StreamHandler()
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Spúšťam trénovanie: Sentence-Embeddings + MLP ...")

        classifier = NewsClassifier(
            use_bert=False,
            use_tfidf=False,
            use_sentence_embeddings=True,
            model_type="mlp",
        )
        classifier.train(enable_augmentation=False)

        logger.info("Trénovanie úspešne dokončené!")

    except Exception as e:
        logger.error(f"Chyba pri trénovaní: {e}")
        raise


if __name__ == "__main__":
    main()
