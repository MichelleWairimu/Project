import asyncio
import gradio as gr
from custom_actions import CustomActions

async def main():
    actions = CustomActions()
    
    # Load data and train the model
    texts, labels = await actions.load_data('project.yaml')
    accuracy = await actions.train_model(texts, labels)
    print(f"Accuracy: {accuracy}")

    # Load the trained model
    model = await actions.load_model('text_classifier_model.pkl')

    # Function to classify PDF using the loaded model
    def classify_pdf(pdf_file):
        # Extract text from PDF
        pdf_text = asyncio.run(actions.extract_text_from_pdf(pdf_file.name))
        # Classify text using the model
        prediction = asyncio.run(actions.classify_text(model, pdf_text))
        return prediction

    # Create Gradio interface
    iface = gr.Interface(
        fn=classify_pdf,
        inputs=gr.File(label="Upload PDF", file_types=[".pdf"]),
        outputs="text",
        title="PDF Genre Categorizer"
    )
    
if __name__ == "__main__":
    asyncio.run(main())
