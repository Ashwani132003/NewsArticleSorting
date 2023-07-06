from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import logging

logging.basicConfig(filename='record.log', level=logging.DEBUG)

model1 = joblib.load('model1.pkl')
vectorizer = joblib.load('pre_fitted_vectorizer.pkl')  # Load the pre-fitted CountVectorizer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', sentences=None, predictions=None)

@app.route('/process1', methods=['POST'])
def process1():
    # Get the input sentences and use model1 for prediction
    sentences = request.form.get('sentences')  # Get the input sentences from the request
    logging.debug("inputs taken")
    
    # Split the sentences into a list
    sentences_list = sentences.split(',')

    # Preprocess the sentences using the imported function
    # processed_sentences = [preprocess_text(sentence) for sentence in sentences_list]

    # Convert the preprocessed sentences into a numeric array
    sentences_array = vectorizer.transform(sentences_list)

    # Use model1 for prediction
    predictions = model1.predict(sentences_array)
    processed_predictions = list(map(lambda x: 'sport' if x==1 else 'business' if x==2 else 'politics' if x==3 else 'entertainment' if x==4 else 'tech', predictions))
    # Return the results using the 'index.html' template
    return render_template('index.html', sentences=sentences_list, predictions=processed_predictions)

if __name__ == '__main__':
    app.run(debug=True)
