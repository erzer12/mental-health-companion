---
title: Mental Health Companion
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: mit
short_description: 'AI-driven chatbot that detects user mood and provides context-aware mental health support'
---

# 🌿 Zen: Mental Health Companion

An **AI-powered mental health support chatbot** designed specifically for students. Zen combines **emotion detection**, **retrieval-augmented generation (RAG)**, and **interactive wellness tools** to provide personalized, empathetic support.

![Gradio](https://img.shields.io/badge/Gradio-6.3.0-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)

---

## ✨ Features

### 🧠 Intelligent Emotion Detection
- Real-time emotion analysis using `j-hartmann/emotion-english-distilroberta-base`
- Detects emotions: joy, sadness, anger, fear, surprise, disgust, neutral
- Adapts responses based on detected emotional state

### 📚 Knowledge-Grounded Responses (RAG)
- Retrieves relevant information from a curated mental health PDF library
- Uses FAISS vector database for fast semantic search
- Smart caching: only rebuilds when documents change

### 💬 Empathetic AI Conversations
- Powered by `Meta-Llama-3-8B-Instruct` via Hugging Face Inference API
- Streaming responses for natural conversation flow
- Context-aware prompts that incorporate emotion + retrieved knowledge

### 🧘 Interactive Wellness Widgets
- **Box Breathing Tool**: Activates for high-arousal emotions (fear, anger, sadness)
- **5-4-3-2-1 Grounding Exercise**: Triggers when panic/overwhelm keywords detected
- Dynamic UI that responds to user's emotional needs

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP A: PERCEPTION                                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Emotion Classifier (DistilRoBERTa)                      │    │
│  │  → Detects: joy, sadness, anger, fear, surprise, etc.   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP B: MEMORY (RAG)                                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  FAISS Vector DB + Sentence Transformers                 │    │
│  │  → Retrieves relevant mental health techniques           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP C: REASONING                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Llama-3-8B-Instruct (Serverless API)                    │    │
│  │  → Generates empathetic, contextual response             │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP D: UI DECISION                                             │
│  → Show breathing widget? Show grounding checklist?              │
│  → Stream response to chat interface                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mental-health-companion/
├── app.py                 # Main Gradio application
├── rag_engine.py          # RAG pipeline (PDF ingestion, FAISS, retrieval)
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/                  # PDF knowledge base (21 mental health resources)
│   ├── Anxiety Information Sheet - 08 - Breathing Retraining.pdf
│   ├── Panic Information Sheet - 01 - What is Panic.pdf
│   ├── Sleep Information Sheet - 04 - Sleep Hygiene.pdf
│   └── ... (18 more PDFs)
└── vectorstore/           # Auto-generated FAISS index (created on first run)
    ├── db_faiss/
    └── manifest.json
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Hugging Face account with API token

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/mental-health-companion
   cd mental-health-companion
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Hugging Face token**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. Open `http://localhost:7860` in your browser

### Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Push this repository to your Space
4. Add `HF_TOKEN` in **Settings → Secrets**

---

## 📚 Knowledge Base

The `/data` folder contains curated mental health resources covering:

| Topic | Resources |
|-------|-----------|
| **Anxiety** | Breathing retraining, progressive muscle relaxation, stress coping |
| **Panic** | Understanding panic attacks, grounding techniques |
| **Procrastination** | Vicious cycle, practical strategies, action planning |
| **Sleep** | Insomnia, sleep hygiene best practices |
| **Social Anxiety** | Understanding social anxiety, breathing exercises |
| **Self-Esteem** | Acknowledging positives, building confidence |
| **Perfectionism** | Challenging perfectionistic thinking |
| **Unhelpful Thinking** | Catastrophising, shoulding/musting, cognitive restructuring |

### Adding New Resources

1. Add PDF files to the `data/` folder
2. Restart the application
3. The RAG engine will automatically detect changes and rebuild the vector database

---

## ⚙️ Configuration

### Models Used

| Component | Model | Provider |
|-----------|-------|----------|
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` | Local (Transformers) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local |
| Chat LLM | `meta-llama/Meta-Llama-3-8B-Instruct` | HF Inference API |

### Customization

- **Change LLM**: Modify `InferenceClient()` in `app.py` line 21
- **Adjust RAG chunks**: Edit `chunk_size` and `chunk_overlap` in `rag_engine.py`
- **Modify widget triggers**: Update emotion conditions in `chat_wrapper()` function

---

## 🛠️ Tech Stack

- **Frontend**: [Gradio](https://gradio.app/) - Modern ML web interfaces
- **Vector DB**: [FAISS](https://github.com/facebookresearch/faiss) - Fast similarity search
- **Embeddings**: [Sentence Transformers](https://sbert.net/) - Semantic text embeddings
- **LLM Orchestration**: [LangChain](https://langchain.com/) - Document loading & text splitting
- **Inference**: [Hugging Face Hub](https://huggingface.co/) - Serverless model APIs

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Mental health resources adapted from [Centre for Clinical Interventions](https://www.cci.health.wa.gov.au/)
- Emotion detection model by [j-hartmann](https://huggingface.co/j-hartmann)
- Built with ❤️ for student mental health
