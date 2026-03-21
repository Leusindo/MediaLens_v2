# training/train_model.py
import logging
import os
import sys
import argparse

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

    parser = argparse.ArgumentParser(description="Trénovanie MediaLens modelu")
    parser.add_argument(
        "--model-type",
        choices=["mlp", "rf"],
        default="mlp",
        help="Typ modelu, ktorý sa má trénovať (mlp alebo rf)."
    )
    args = parser.parse_args()

    try:
        logger.info(f"Spúšťam trénovanie modelu typu: {args.model_type}")

        classifier = NewsClassifier(model_type=args.model_type)
        classifier.train()

        logger.info("Trénovanie úspešne dokončené!")

    except Exception as e:
        logger.error(f"Chyba pri trénovaní: {e}")
        raise


if __name__ == "__main__":
    main()
