# custom_actions.py
import yaml
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import asyncflows

class CustomActions(asyncflows.Action):
    name = "CustomActions"

    def __init__(self, temp_dir=None, log=None):
        super().__init__(temp_dir=temp_dir, log=log)

    async def load_data(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)  # Load YAML data using PyYAML
        texts = [entry['text'] for entry in data]
        labels = [entry['category'] for entry in data]
        return texts, labels

    async def train_model(self, texts, labels):
        train_docs, test_docs, train_labels, test_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(train_docs, train_labels)
        joblib.dump(model, 'text_classifier_model.pkl')
        predicted_labels = model.predict(test_docs)
        accuracy = metrics.accuracy_score(test_labels, predicted_labels)
        return accuracy

    async def load_model(self, model_path):
        return joblib.load(model_path)

    async def classify_text(self, model, text):
        return model.predict([text])[0]

    async def extract_text_from_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
