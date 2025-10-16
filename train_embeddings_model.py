import nltk
import random
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the KG Reddy College intents data
with open('dataset/kgreddy_intents.json', 'r') as f:
    intents = json.load(f)

# Initialize the sentence transformer model
model_name = 'paraphrase-MiniLM-L6-v2'
print(f"Loading sentence transformer model: {model_name}")
sentence_model = SentenceTransformer(model_name)
print(f"Loaded sentence transformer model: {model_name}")

# Prepare training data
X = []
y = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        X.append(pattern)
        y.append(intent['tag'])

print(f"Total training examples: {len(X)}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

# Generate embeddings for training data
print("Generating embeddings for training data...")
X_train_embeddings = sentence_model.encode(X_train)
X_test_embeddings = sentence_model.encode(X_test)

print(f"Embedding dimension: {X_train_embeddings.shape[1]}")

# Train a classifier on the embeddings
print("Training classifier...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_embeddings, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_embeddings)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Test with college-specific paraphrases
test_queries = [
    "tell me about college fees",
    "what are the tuition charges?",
    "how much does it cost to study at this college?",
    "where is the college located?",
    "what's the location of your college campus?",
    "how can I find your college campus?",
    "what courses does the college offer?",
    "tell me about admission process",
    "what are the hostel facilities?"
]

# Generate embeddings for test queries
print("\nTesting with paraphrases:")
test_embeddings = sentence_model.encode(test_queries)
predictions = classifier.predict(test_embeddings)

# Display results
for query, prediction in zip(test_queries, predictions):
    print(f"Query: '{query}' → Predicted intent: '{prediction}'")

# Generate embeddings for all training data for the final model
print("\nTraining final model on all data...")
all_embeddings = sentence_model.encode(X)

# Train the final model on all data
final_classifier = LogisticRegression(max_iter=1000)
final_classifier.fit(all_embeddings, y)

# Save the model and sentence transformer
print("Saving model and sentence transformer...")
with open('model/chatbot_model_embeddings.pkl', 'wb') as f:
    pickle.dump(final_classifier, f)

# Save the sentence transformer model name
with open('model/sentence_transformer_model.pkl', 'wb') as f:
    pickle.dump(model_name, f)

print("Model and sentence transformer saved successfully!")