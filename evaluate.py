import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(filename='parse_ner.log', level=logging.INFO)

# Load model and tokenizer
model_path = "./models/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

def predict_entities(text):
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # Convert predictions to tags
    predicted_tokens = []
    for token, pred in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), predictions[0]):
        if not token.startswith("##"):
            predicted_tokens.append((token, model.config.id2label[pred.item()]))
    
    return predicted_tokens

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        predictions = predict_entities(text)
        return jsonify({"predictions": predictions})
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7707)
