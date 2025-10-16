# College Chatbot Using ML Algorithm and NLP Toolkit 

The College Chatbot(for kg reddy college) is a Python-based chatbot that utilizes machine learning algorithms and natural language processing (NLP) techniques to provide automated assistance to users with college-related inquiries. The chatbot aims to improve the user experience by delivering quick and accurate responses to their questions.

## Features

- **Intent-Based Responses**: Accurately identifies user questions and provides relevant information
- **Professional UI**: Clean, responsive dark-themed interface with message bubbles
- **Confidence Threshold**: Prevents incorrect answers by using a 0.6 confidence threshold
- **Context Memory**: Maintains conversation history for natural multi-turn interactions
- **Automatic Link Detection**: Converts URLs in responses to clickable links
- **Fallback Mechanism**: Suggests possible topics when queries are unclear

## Methodology
The chatbot is developed using a combination of natural language processing techniques and machine learning algorithms. The methodology involves data preparation, model training, and chatbot response generation. The data is preprocessed to remove noise and increase training examples using synonym replacement. Multiple classification models are trained and evaluated to find the best-performing one. The trained model is then used to predict the intent of user input, and a random response is selected from the corresponding intent's responses. The chatbot is developed as a web application using Flask, allowing users to interact with it in real-time.


## Technical Details

### Architecture

- **Frontend**: HTML, CSS, JavaScript with AJAX for asynchronous communication
- **Backend**: Flask web server
- **NLP Model**: TF-IDF vectorization with machine learning classification
- **Alternative Model**: Sentence embeddings for improved semantic understanding
- **Context Management**: Session-based memory for multi-turn conversations
- **Confidence System**: Threshold-based response filtering with fallback suggestions

### Files Structure

- `app.py`: Main Flask application with routing and chatbot logic
- `templates/index.html`: Frontend interface template
- `static/styles.css`: CSS styling for the chatbot UI
- `dataset/`: Contains JSON intent files with training data
- `model/`: Stores trained ML models and vectorizers
- `train_embeddings_model.py`: Script to train the sentence embeddings model

## Motivation
The motivation behind this project was to create a simple chatbot using my newly acquired knowledge of Natural Language Processing (NLP) and Python programming. As one of my first projects in this field, I wanted to put my skills to the test and see what I could create.

[I followed a guide referenced in the project](https://thecleverprogrammer.com/2023/03/27/end-to-end-chatbot-using-python/) to learn the steps involved in creating an end-to-end chatbot. This included collecting data, choosing programming languages and NLP tools, training the chatbot, and testing and refining it before making it available to users.

Although this chatbot may not have exceptional cognitive skills or be state-of-the-art, it was a great way for me to apply my skills and learn more about NLP and chatbot development. I hope this project inspires others to try their hand at creating their own chatbots and further explore the world of NLP.


```

## Usage Examples

The chatbot can answer questions about:
- College programs and courses
- Fee structures
- Placement statistics
- Admission procedures
- Campus facilities

Example interactions:
- "What programs does the college offer?"
- "Tell me about CSE placements"
- "What are the fees for B.Tech?"
- "And what about its placements?" (demonstrates context memory)









