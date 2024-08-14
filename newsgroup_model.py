import joblib, os
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset"
DATASET_DIR.mkdir(exist_ok=True, parents=True)

# Load some categories of newsgroups dataset
categories = [
    "talk.politics.guns",
    'talk.politics.mideast',
    'talk.politics.misc',
    "talk.religion.misc",
    "comp.sys.mac.hardware",
    "sci.crypt",
]
newsgroups_training = fetch_20newsgroups(
    subset="train", categories=categories, random_state=0, data_home=DATASET_DIR
)
newsgroups_testing = fetch_20newsgroups(
    subset="test", categories=categories, random_state=0, data_home=DATASET_DIR
)

# Make the pipeline
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB(),
)

# Train the model
model.fit(newsgroups_training.data, newsgroups_training.target)

# Serialize the model and the target names
model_file = DATASET_DIR / "newsgroups_model.joblib"
model_targets_tuple = (model, newsgroups_training.target_names)
joblib.dump(model_targets_tuple, model_file)
