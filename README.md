# 🌿 Zen: Mental Health Companion

An **AI-powered mental health support chatbot** for students. Zen combines **emotion detection**, **RAG-powered knowledge retrieval**, and **interactive wellness tools** to provide personalized, empathetic support.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow?style=for-the-badge)](https://huggingface.co/spaces/Erzer12/mental-health-companion)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/erzer12/mental-health-companion)

![Gradio](https://img.shields.io/badge/Gradio-6.3.0-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)

---

## ✨ Features

### 🧠 Enhanced Emotion Detection
- Real-time emotion analysis using `j-hartmann/emotion-english-distilroberta-base`
- **Smart overrides**: Academic stress → `stress` (not sadness), panic keywords → `panic`
- Detects: joy, sadness, anger, fear, stress, panic, anxiety, dissociation, neutral

### 📚 Knowledge-Grounded Responses (RAG)
- Retrieves from 21 curated mental health PDFs
- FAISS vector database with smart caching
- Topics: anxiety, panic, procrastination, sleep, self-esteem, cognitive techniques

### 💬 Empathetic AI Conversations
- Powered by `Meta-Llama-3-8B-Instruct` via Hugging Face Inference API
- Natural, friend-like tone (never clinical or robotic)
- Context-aware prompts incorporating emotion + retrieved knowledge

### 🧘 Interactive Wellness Widgets

| Widget | Trigger | Description |
|--------|---------|-------------|
| **🌬️ Animated Breathing** | panic, anxiety, fear | CSS-animated circle for box breathing (4-4-4-4) |
| **🦶 5-4-3-2-1 Grounding** | dissociation, panic | Interactive checklist for sensory grounding |

### 🎨 Dark Glassmorphism UI
- Calming dark theme with emerald accents
- Backdrop blur effects
- Smooth hover animations

---

## 🏗️ Architecture

```
USER INPUT
    │
    ▼
┌─────────────────────────────────────────┐
│  STEP A: ENHANCED PERCEPTION            │
│  • ML Emotion Classifier                │
│  • Keyword-based overrides (panic,      │
│    procrastination, dissociation)       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  STEP B: MEMORY (RAG)                   │
│  • FAISS Vector DB                      │
│  • Sentence Transformers embeddings     │
│  • 21 mental health PDFs indexed        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  STEP C: REASONING                      │
│  • Llama-3-8B-Instruct                  │
│  • Dynamic system prompt with context   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  STEP D: UI DECISION                    │
│  • Show breathing widget?               │
│  • Show grounding checklist?            │
│  • Stream response                      │
└─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
mental-health-companion/
├── app.py                 # Main Gradio app (UI + logic)
├── rag_engine.py          # RAG pipeline (PDF ingestion, FAISS)
├── requirements.txt       # Dependencies (pinned versions)
├── README.md              # This file
├── .gitignore             # Excludes vectorstore/, __pycache__/
├── data/                  # 21 mental health PDFs
│   ├── Anxiety Information Sheet - *.pdf
│   ├── Panic Information Sheet - *.pdf
│   └── ...
├── vectorstore/           # Auto-generated on first run
└── .github/workflows/     # CI/CD
    └── sync_to_hub.yml    # Auto-sync to HF Spaces
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Hugging Face token ([get one here](https://huggingface.co/settings/tokens))

### Local Development

```bash
# Clone
git clone https://github.com/erzer12/mental-health-companion.git
cd mental-health-companion

# Install
pip install -r requirements.txt

# Set token
export HF_TOKEN="your_huggingface_token"

# Run
python app.py
```

Open `http://localhost:7860`

### Deploy to Hugging Face Spaces

1. Fork/push to a new HF Space (Gradio SDK)
2. Add `HF_TOKEN` secret in Space settings

---

## ⚙️ Configuration

### Models

| Component | Model | Provider |
|-----------|-------|----------|
| Emotion Detection | `j-hartmann/emotion-english-distilroberta-base` | Local |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local |
| Chat LLM | `meta-llama/Meta-Llama-3-8B-Instruct` | HF Inference API |

### Emotion Override Rules

| Keywords | Classified As |
|----------|---------------|
| homework, exam, lazy, procrastinating | `stress` |
| panic, heart racing, can't breathe | `panic` |
| anxious, worried, scared | `anxiety` |
| unreal, numb, disconnected | `dissociation` |

---

## 🛠️ Tech Stack

- **Frontend**: Gradio 6.3.0 with custom CSS
- **Vector DB**: FAISS
- **Embeddings**: Sentence Transformers
- **LLM**: LangChain + Hugging Face Inference
- **CI/CD**: GitHub Actions → HF Spaces sync

---

## 📝 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- Mental health resources from [Centre for Clinical Interventions](https://www.cci.health.wa.gov.au/)
- Emotion model by [j-hartmann](https://huggingface.co/j-hartmann)
- Built with ❤️ for student mental health
