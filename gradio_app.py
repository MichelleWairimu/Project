import joblib
import fitz  # PyMuPDF
import gradio as gr

# Load the pre-trained model
model = joblib.load('text_classifier_model.pkl')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def classify_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file.name)
    return model.predict([text])[0]

iface = gr.Interface(fn=classify_pdf, inputs=gr.components.File(file_types=['.pdf']), outputs="text", title="PDF Categorizer")
iface.launch()
