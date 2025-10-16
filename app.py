from flask import Flask, render_template, request, session
import pickle
import json
import random
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = 'kgreddy_chatbot_secret_key'  # Required for session

# Load the trained model and sentence transformer model name
with open('model/chatbot_model_embeddings.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('model/sentence_transformer_model.pkl', 'rb') as f:
    model_name = pickle.load(f)

# Initialize the sentence transformer
sentence_model = SentenceTransformer(model_name)

# Load the KG Reddy College intents data
with open('dataset/kgreddy_intents.json', 'r') as f:
    intents = json.load(f)

# Get the top 3 intents for fallback suggestions
def get_top_intents(intent_probs, n=3):
    # Get indices of top n probabilities
    top_indices = intent_probs.argsort()[-n:][::-1]
    top_intents = [best_model.classes_[i] for i in top_indices]
    
    # Get a sample pattern for each intent to suggest
    suggestions = []
    for intent_tag in top_intents:
        for intent in intents['intents']:
            if intent['tag'] == intent_tag and len(intent['patterns']) > 0:
                suggestions.append(random.choice(intent['patterns']))
                break
    
    return suggestions

def chatbot_response(user_input):
    # Initialize session context if it doesn't exist
    if 'context' not in session:
        session['context'] = []
    
    # Add current input to context
    session['context'].append(user_input)
    
    # Use the last 3 messages for context (or fewer if not available)
    context_window = session['context'][-3:]
    combined_input = " ".join(context_window)
    
    # Generate embedding for combined input (with context)
    input_embedding = sentence_model.encode([combined_input])
    
    # For fallback, use the default intent
    predicted_intent = "fallback"
    
    # Try to find the best matching intent
    intent_probs = best_model.predict_proba(input_embedding)[0]
    best_intent_idx = intent_probs.argmax()
    max_confidence = intent_probs[best_intent_idx]
    
    # Check confidence threshold (0.6)
    if max_confidence >= 0.6:
        predicted_intent = best_model.classes_[best_intent_idx]
        # Find matching intent and return KG Reddy College-specific response
        for intent in intents['intents']:
            if intent['tag'] == predicted_intent:
                response = random.choice(intent['responses'])
                break
    else:
        # Fallback with suggestions
        suggestions = get_top_intents(intent_probs)
        if suggestions:
            response = f"I'm not sure I understood. Did you mean one of these?\n- {suggestions[0]}"
            if len(suggestions) > 1:
                response += f"\n- {suggestions[1]}"
            if len(suggestions) > 2:
                response += f"\n- {suggestions[2]}"
        else:
            response = "I'm not sure I understood. Try rephrasing your question!"
            
    return response

@app.route('/')
def home():
    # Clear session when loading home page
    session.clear()
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001)