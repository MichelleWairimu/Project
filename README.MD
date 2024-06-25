The Code categorizes pdf into various categories. The categories can be changed in the project.yaml file to fit the users needs.

The code can be reused by traiing the model and changing the categories it classifies data to icrease time efficiency and improved organisation

To setup your development environment
        Create your environment and activate it by running the following in the terminal

                python -m venv .venv
                source .venv/Scripts/activate

        Run the following in the terminal to install the various libraries

                pip install -r requirements.txt

To run the Project Run the following in the terminal

                python gradio_app.py

CUSTOM ACTIONS

The load_data function extracts code from the .yaml file.

The train_model function then trains a text classification model using the provided texts and labels. the trained model is saved to the text_classifier_model.pkl and it returns accuracy of the model

The load_model function the loads the pretrained model from the text_classifier_model.pkl file

The extract_text_from_pdf is called with a path to the pdf and text from each page is extracted and returned

The returned text is then classified under the classify_text function and the categories returned

FLOW

In the main.py:
        An instance for the CustomActions class is created

        The load_data function is called to load the data from the .yaml file

        The train_model function is called to train the model and the accuracy is returned

        The load_model function is called to load the model from a file

        The classify_pdf function extracts text from the uploaded pdf, the loaded model is used to classify the extracted text and the prediction is returned

        A gradio interface is created that expects the input to be a pdf. It,s output is a text which is the predicted genre