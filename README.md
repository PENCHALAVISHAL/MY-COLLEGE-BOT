### Project summary

**Core Functionality:**
- **College Information Assistant**: Answers questions about KG Reddy College of Engineering & Technology (KGRCET)
- **Web-based Chat Interface**: Modern, responsive UI accessible at `http://127.0.0.1:5050`
- **Multi-layered Response System**: Uses 6 different strategies to provide accurate answers

**What It Can Answer:**

1. **College Overview & Identity**
   - College name, location, establishment details
   - General information about KGRCET

2. **Academic Programs**
   - B.Tech courses (CSE, CSE-AI/ML, CSE-DS, ECE, EEE, Mechanical, Civil)
   - M.Tech, MBA, MCA programs
   - Department information and intake details

3. **Admissions Process**
   - TS EAMCET admission process (EAMCET code: KGRH)
   - Management quota information
   - Application procedures

4. **Financial Information**
   - Fee structure and payment details
   - Tuition costs (approximately ₹90,000–₹1,03,000/year for B.Tech)

5. **Campus Life & Facilities**
   - Hostel facilities (boys' hostel on campus, girls' hostel at Mehdipatnam)
   - Library resources (~25,700+ volumes, e-journals)
   - Transport services and bus routes
   - Campus facilities and infrastructure

6. **Career & Placements**
   - Placement statistics and recruiter information
   - Highest packages (up to ~₹16 LPA)
   - Career services and opportunities

7. **Contact & Support**
   - College address, phone numbers, email
   - Office hours and contact methods
   - Location and directions

**How It Works:**
- **Smart Response Selection**: Tries multiple methods to find the best answer
- **Official Links**: Provides direct links to college website sections
- **Fallback Safety**: Always provides helpful responses even for unknown queries
- **Real-time Interaction**: Instant responses through web interface

**User Experience:**
- Clean, modern chat interface with message bubbles
- Quick reply buttons for common topics
- Mobile-responsive design
- Safe error handling - never crashes or gives confusing responses

The chatbot essentially serves as a **24/7 virtual college counselor** that can instantly answer most common questions prospective students, current students, or parents might have about KGRCET.

### Key Features
- ML intent classification using a trained `scikit-learn` model (`model/chatbot_model.pkl`) and `vectorizer.pkl` for text features
- Structured intents dataset (`dataset/intents1.json`) with tags, patterns, responses for deterministic replies
- College knowledge base (`dataset/kgr_kb.json`) with keywords → canonical answers (includes official links)
- Optional semantic retrieval with `sentence-transformers` for meaning-based matches (offline, no internet required after first model load)
- Optional LLM responses grounded by the knowledge base
  - Cloud: OpenAI (if `OPENAI_API_KEY` set)
  - Local: Ollama (if running a local model and `OLLAMA_BASE_URL` configured)
- Simple, modern chat UI (`templates/index.html`, `static/styles.css`)


## Architecture Overview

```

### Request Flow
1. UI posts `user_input` to `/chat`.
2. `chatbot_response(user_input)` applies a cascade of strategies with safe fallbacks:
   - Local Ollama → Cloud OpenAI → Semantic retrieval → Fuzzy match → Exact KB → ML classifier
3. Returns a short, safe response (HTML allowed for links). When confidence is low, returns a helpful default prompt.

---

## Project Structure



---

## Setup

### Prerequisites
- Python 3.10–3.12 recommended
- Windows/macOS/Linux

### Create and activate a virtual environment
```
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux
```

### Install dependencies
```
pip install -r requirements.txt
```

Notes:
- `sentence-transformers` and `torch` are optional; the app runs without them (semantic retrieval will be skipped). On Windows with Python ≥3.12 you may need to install a compatible `torch` or skip semantic mode.
- OpenAI and Ollama integrations are optional.

### Run the app
```
python app.py
```
Open `http://127.0.0.1:5050` in your browser.

---

## Configuration

Environment variables (optional):

```
# OpenAI (cloud LLM)
OPENAI_API_KEY=sk-...            # enables OpenAI path
OPENAI_MODEL=gpt-4o-mini         # optional, default: gpt-4o-mini

# Ollama (local LLM)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=phi4                # any installed Ollama chat model

# Sentence-Transformers (semantic retrieval)
SBERT_MODEL=all-MiniLM-L6-v2     # optional, default: all-MiniLM-L6-v2
```

Place these in a `.env` file or export them in your shell. The app auto-loads `.env` if `python-dotenv` is installed.

---

## Data Formats

### Intents (`dataset/intents1.json`)
Minimal schema:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello"],
      "responses": ["Hello!", "Hi there, how can I help?"],
      "context_set": ""
    }
  ]
}
```

Usage:
- During ML fallback, predicted `tag` selects a random response from the matching intent.

### Knowledge Base (`dataset/kgr_kb.json`)
Minimal schema:

```json
{
  "meta": {
    "name": "...",
    "short": "...",
    "website": "...",
    "last_updated": "YYYY-MM-DD"
  },
  "links": { "home": "...", "admissions": "..." },
  "intents": [
    {
      "tag": "courses_list",
      "keywords": ["courses", "branches"],
      "response": "Programs: ... <a href=\"...\">Academics</a>"
    }
  ]
}
```

Resolution order references this KB for: semantic retrieval, fuzzy match, and exact keyword match.

---

## How Responses Are Generated

`chatbot_response(user_input)` performs:
1. Input sanitization and empty-check
2. Optional LLM calls (Ollama → OpenAI) with a system prompt grounded in `kgr_kb.json`
3. Optional semantic retrieval: cosine similarity over SBERT embeddings of KB responses
4. Fuzzy keyword matching with RapidFuzz to tolerate minor wording changes
5. Exact keyword match against KB intents
6. ML intent classification using `vectorizer.pkl` + `chatbot_model.pkl`, then map `tag` → responses from `intents1.json`

If confidence is low or nothing matches, a safe default hint is returned asking the user to rephrase or try supported topics.

---

## Training the ML Model (overview)

The repository includes pre-trained artifacts in `model/`. If you want to retrain:
- Prepare/expand `dataset/intents1.json` patterns per tag
- Tokenize/clean text, generate features (e.g., TF‑IDF)
- Train a classifier (e.g., Logistic Regression, SVM, etc.)
- Save artifacts as `chatbot_model.pkl` and `vectorizer.pkl` (must match runtime code)

You can prototype training in `Chatbot.ipynb` / `College Chatbot.ipynb` or your own notebook/script.

---

## Running Locally with Ollama (optional)
1. Install Ollama from `https://ollama.com/` and start the daemon
2. Pull a chat model, e.g.: `ollama pull phi4`
3. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` env vars
4. Start the app and chat

The app will prefer Ollama responses and fall back to other strategies if a local reply fails.

---

## Security & Reliability Considerations
- Inputs are normalized and sanitized for matching; backend returns plain text or simple HTML links
- Errors are caught and a generic, user‑friendly message is returned so the UI stays responsive
- LLM usage is optional; without keys or local models, the app still works via KB + ML

---

## Troubleshooting
- "Could not reach the server": ensure the Flask app is running on port 5050
- OpenAI errors: verify `OPENAI_API_KEY` and network access
- Ollama errors: verify daemon is running and model is available
- `torch` install issues on Windows: skip semantic retrieval or install a compatible CPU wheel

---

## License
This project is for educational purposes. Add your preferred license here (e.g., MIT) if you plan to distribute.

---

## Acknowledgments
- `Flask`, `scikit-learn`, `sentence-transformers`, `rapidfuzz`
- Public college website data referenced via links stored in the KB
