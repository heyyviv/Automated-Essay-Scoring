from flask import Flask, request, render_template
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load the model and tokenizer from the pickle file
with open('model.pkl', 'rb') as f:
    tokenizer, model = pickle.load(f)

MAX_LENGTH = 1024

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    if request.method == 'POST':
        input_text = request.form['essay']

        # Tokenize input text
        inputs = tokenizer(input_text, max_length=MAX_LENGTH, truncation=True, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits[0]).item() + 1

        return render_template('index.html', score=predicted_class, essay=input_text)

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
