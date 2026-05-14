# 🪶 Thoth AI — Continuously Learning Text Generation Engine

<p align="center">
  <img src="static/logo.png" width="120" alt="Thoth AI Logo">
</p>

<p align="center">
  <strong>A self-learning AI chatbot that grows smarter with every conversation.</strong><br>
  Built with a custom Transformer architecture, dynamic tokenizer expansion, and real-time training.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-Thoth%20V1-00D4FF?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgo" alt="Version">
  <img src="https://img.shields.io/badge/Architecture-Transformer-gold?style=for-the-badge" alt="Architecture">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%2FKeras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
</p>

---

## ✨ Features

### 🧠 Continuous Learning Pipeline
- **Every message trains the model** — The model runs 30 epochs of training on each user input in the background
- **Dynamic vocabulary expansion** — New words are automatically added to the BPE tokenizer without losing existing knowledge
- **Dynamic model resizing** — Embedding and output layers expand automatically when vocabulary grows, preserving all trained weights

### 🎨 ChatGPT-Style Interface
- Premium dark theme with glassmorphism effects
- Typewriter animation for AI responses
- Quick-start prompt chips on the welcome screen
- Mobile-responsive sidebar layout

### 📊 Model Dashboard
- Real-time model statistics: parameter count, vocabulary size, context window, training sessions
- Live training log viewer with auto-polling
- Visual training history with loss tracking

### 🔧 Advanced Features
- **Bulk Train** — Paste large paragraphs or articles to train the model on massive text at once
- **Export Chat** — Download your entire conversation as a `.txt` file
- **Keyboard Shortcuts** — `Ctrl+Shift+N` for new chat, `Escape` to close panels
- **Temperature Control** — Adjust creativity from 0.1 (focused) to 2.0 (wild)
- **Anti-Repetition Engine** — Hard token banning prevents repetitive output loops
- **Thread-Safe Training** — Concurrent requests handled safely with locking
- **Crash-Proof Saving** — Atomic file saves prevent weight corruption

---

## 🏗️ Architecture

```
Thoth V1 — Custom MiniGPT Transformer
├── Token Embedding    (vocab_size × 256)
├── Position Embedding (512 × 256)         ← Supports 512-token context
├── Transformer Block × 6
│   ├── Multi-Head Self-Attention (4 heads, causal mask)
│   ├── Layer Normalization
│   ├── Feed-Forward Network (256 → 1024 → 256, GELU)
│   └── Residual Connections
├── Final Layer Norm
└── Linear Head → vocab_size (logits)
```

| Spec | Value |
|------|-------|
| Parameters | ~10M |
| Layers | 6 |
| Embedding Dim | 256 |
| Attention Heads | 4 |
| Context Window | 512 tokens |
| Tokenizer | BPE (HuggingFace Tokenizers) |
| Training | 30 epochs per input, Adam (lr=3e-4) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Thoth-AI.git
cd Thoth-AI

# Install dependencies
pip install tensorflow keras fastapi uvicorn tokenizers

# Run the server
python app.py
```

Open **http://localhost:8001** in your browser.

### Bulk Training (Optional)

Feed the model large amounts of text to improve quality:

```bash
python bulk_train.py
```

Or use the **Bulk Train** button in the sidebar to paste text directly.

---

## 📁 Project Structure

```
Thoth-AI/
├── app.py                          # FastAPI server with all API endpoints
├── model_utils.py                  # Core model architecture, training, generation
├── bulk_train.py                   # Script to bulk-train with large paragraphs
├── train_tokenizer.py              # BPE tokenizer training script
├── minigpt.keras                   # Saved model weights (Keras format)
├── minigpt_tokenizer (1).json      # BPE tokenizer vocabulary
├── training_log.txt                # Training session history
├── user_dataset.txt                # All user inputs saved for reference
├── static/
│   ├── index.html                  # ChatGPT-style frontend
│   ├── style.css                   # Premium dark theme CSS
│   ├── script.js                   # Frontend logic & interactions
│   └── logo.png                    # Thoth ibis logo
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the web interface |
| `POST` | `/generate` | Generate text from a prompt |
| `POST` | `/train` | Train model on custom text |
| `GET` | `/logs` | Retrieve training log history |
| `GET` | `/stats` | Get model statistics (params, vocab, etc.) |

### Example: Generate Text

```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The sun rises", "temperature": 0.8}'
```

### Example: Train on Custom Text

```bash
curl -X POST http://localhost:8001/train \
  -H "Content-Type: application/json" \
  -d '{"text": "Your training paragraph goes here..."}'
```

---

## 🧪 How It Works

1. **User sends a message** → Frontend sends prompt to `/generate`
2. **Model generates response** → Token-by-token with temperature sampling + repetition penalty
3. **Background training fires** → The user's prompt is used to train the model (30 epochs)
4. **Tokenizer checks for new words** → Unknown words are added to vocabulary
5. **Model resizes if needed** → Embedding layers expand to accommodate new tokens
6. **Weights saved atomically** → Crash-proof save to prevent corruption

---

## 📝 Why Named "Thoth"?

**Thoth** (𓅝) is the ancient Egyptian god of wisdom, writing, hieroglyphics, science, and knowledge. He was believed to have **invented writing itself** and served as the scribe of the gods. As an AI that continuously learns to write better text, Thoth is the perfect namesake.

---

## ⚠️ Limitations

- This is a **~10M parameter model** — significantly smaller than production models like GPT-2 (117M) or GPT-3 (175B)
- Output quality improves with more training data — feed it books, articles, and paragraphs
- The model learns best from **English text**
- Training happens on CPU, so large texts may take time

---

## 📄 License

MIT License — Feel free to use, modify, and distribute.

---

<p align="center">
  <strong>Built with 🪶 by the Thoth AI project</strong>
</p>
