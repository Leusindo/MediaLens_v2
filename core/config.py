# core/config.py
import os

import torch


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'training_data.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'data', 'models')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')


    BERT_MODEL_NAME = "gerulata/slovakbert"
    MAX_LENGTH = 128
    BATCH_SIZE = 4

    USE_BERT = False
    USE_TFIDF = False
    BERT_EMBEDDING_DIM = 768
    REDUCED_BERT_DIM = 100
    TFIDF_MAX_FEATURES = 2000

    # Sentence embeddings (Sentence-Transformers)
    USE_SENTENCE_EMBEDDINGS = True
    SENTENCE_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    SENTENCE_EMBEDDING_DIM = 384


    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 100

    # Model type: 'rf' or 'mlp'
    MODEL_TYPE = 'mlp'


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CATEGORIES = [
        'clickbait',
        'conspiracy',
        'false_news',
        'propaganda',
        'satire',
        'misleading',
        'biased',
        'legitimate'
    ]