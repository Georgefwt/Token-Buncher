import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flask import Flask, request, jsonify
import os


class ClassifierRewardModel(nn.Module):
    def __init__(self, classifier_model_name, device):
        super().__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
        self.device = device
        self.classifier.to(self.device)
        self.classifier.eval() # Ensure the classifier is in eval mode
    
    @torch.no_grad()
    def score_texts(self, texts):
        """Score a batch of texts directly"""
        batch_size = len(texts)
        rewards = []
        token_lengths = []
        for text in texts:
            # Tokenize the text for the DeBERTa classifier
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            if inputs['input_ids'].shape[1] == 1024:
                print(f"Max length 1024 reached")

            # Get classifier logits
            outputs = self.classifier(**inputs)
            logits = outputs.logits
            harmful_score = -logits[0, 1].item()  # Negative score for attack purpose
            rewards.append(harmful_score)
            token_lengths.append(inputs['input_ids'].shape[1])

        return {
            "rewards": rewards,
            "token_lengths": token_lengths
        }


# Flask application
app = Flask(__name__)

# Global model variable
model = None

def initialize_model(classifier_model_name):
    """Initialize the model with the specified parameters"""
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassifierRewardModel(classifier_model_name, device)
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/score', methods=['POST'])
def score_texts():
    """Score a batch of texts"""

    data = request.json
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing 'texts' field in request"}), 400
    
    texts = data['texts']
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    
    scores = model.score_texts(texts)
    
    return jsonify({
        "scores": scores["rewards"],
        "token_lengths": scores["token_lengths"],
        "count": len(scores["rewards"])
    })

if __name__ == "__main__":
    # python scripts/host_deberta_api.py
    # Get model configuration from environment variables or use defaults
    classifier_model = "domenicrosati/deberta-v3-xsmall-beavertails-harmful-qa-classifier"
    port = int(os.environ.get("PORT", 50050))
    
    # Initialize the model
    initialize_model(classifier_model)
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=port)