# training/train_model.py
import logging
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from core.classifier import NewsClassifier



def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Spúšťam trénovanie modelu...")

        classifier = NewsClassifier()
        classifier.train()

        logger.info("Trénovanie úspešne dokončené!")

    except Exception as e:
        logger.error(f"Chyba pri trénovaní: {e}")
        raise


if __name__ == "__main__":
    main()
