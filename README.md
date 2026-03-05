# 🔥 HappyTorch
**🚀 News: The plugin [happytorch-plugin](https://github.com/Rivflyyy/happytorch-plugin) has been released.**

**A PyTorch coding practice platform — covering LLM, Diffusion, PEFT, and more**

A friendly environment to help you deeply understand deep learning components through hands-on practice.

*Like LeetCode, but for tensors. Self-hosted. Supports both Jupyter and Web interfaces. Instant feedback.*

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

![Problems](https://img.shields.io/badge/problems-24-orange?style=flat-square)
![GPU](https://img.shields.io/badge/GPU-not%20required-brightgreen?style=flat-square)

---

## 🙏 Acknowledgments

This project is a derivative work based on the excellent [TorchCode](https://github.com/duoan/TorchCode) by [@duoan](https://github.com/duoan).

**HappyTorch** extends the original project with **11 additional problems** covering:
- Modern activation functions (GELU, SiLU, SwiGLU)
- Parameter-efficient fine-tuning (LoRA, DoRA)
- Conditional modulation for Diffusion models (AdaLN, AdaLN-Zero, FiLM)
- LLM inference components (RoPE, KV Cache)
- Diffusion training schedules (Sigmoid Noise Schedule)

We sincerely thank the original author for creating such a well-designed practice platform. If you find this project helpful, please also give a ⭐ to the [original repository](https://github.com/duoan/TorchCode).


---

## 🎯 Why HappyTorch?

If you're learning deep learning or preparing for machine learning interviews, you might have encountered these challenges:

- You've read many papers, but don't know where to start when it comes to actually writing code
- You're asked to implement `softmax` or `MultiHeadAttention` from scratch in an interview, and your mind goes blank
- You want to deeply understand technologies like Transformer, LoRA, Diffusion, but feel like you lack systematic practice

**HappyTorch** aims to provide a friendly practice environment. We believe:

> "True understanding comes from hands-on practice."

You can only truly grasp the details of these algorithms by implementing them yourself. This project brings together 24 carefully selected problems, from basic activation functions to complex Transformer components, helping you improve step by step.

### Is this project for you?

- ✅ You're a deep learning beginner looking to solidify your fundamentals
- ✅ You're preparing for ML/AI related interviews
- ✅ You want to understand PyTorch internals
- ✅ You're interested in cutting-edge technologies like LLM, Diffusion, PEFT

### What we offer

| | Feature | |
|---|---|---|
| 🧩 | **24 curated problems** | From basics to advanced, covering mainstream tech stacks |
| ⚖️ | **Auto-grading** | Instant feedback showing what you got right and where to improve |
| 🎨 | **Clear test results** | Each test case displayed separately for easy debugging |
| 💡 | **Helpful hints** | Get nudges when stuck, not full spoilers |
| 📖 | **Reference solutions** | Compare and learn after your own attempt |
| 📊 | **Progress tracking** | Record your learning journey |
| 🌐 | **Web interface** | LeetCode-like practice experience |

### A gentle reminder

This project is **not** a shortcut to pass interviews, nor does it teach you "how to ace interviews." It's simply a practice tool. Real progress comes from your own thinking and hands-on practice.

If you find this project useful, a Star ⭐ would be appreciated. Feel free to share it with others who are learning. Let's grow together!


---

## 🚀 Quick Start

```bash
# 1. Create and activate environment
conda create -n torchcode python=3.11 -y
conda activate torchcode

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyterlab numpy
pip install -e .

# 3. Prepare notebooks
python prepare_notebooks.py

# 4a. Launch Web Mode (recommended for beginners)
pip install fastapi uvicorn python-multipart
python start_web.py
# Open http://localhost:8000

# 4b. Or launch Jupyter Mode (for more flexibility)
python start_jupyter.py
# Open http://localhost:8888
```

---

## 🌐 Web Mode (LeetCode-like Interface)

HappyTorch now offers a web-based practice interface similar to LeetCode!

### Features

- **Monaco Editor** — VS Code's editor with Python syntax highlighting
- **Random Mode** — Get random unsolved problems
- **Sequential Mode** — Work through problems in order
- **Instant Testing** — Run tests with one click (Ctrl+Enter)
- **Progress Dashboard** — Track your solved/attempted/todo status
- **Dark Theme** — Modern, eye-friendly interface

### Installation

```bash
# Install additional web dependencies
pip install fastapi uvicorn python-multipart
```

### Launch

```bash
# Start the web server
python start_web.py
```

Then open **http://localhost:8000** in your browser.

### Screenshots

The web interface includes:
- **Sidebar** — Filterable problem list with difficulty badges and status
- **Code Editor** — Write your implementation with syntax highlighting
- **Description Panel** — View problem details and hints
- **Results Panel** — See test results with pass/fail status and timing

---

## 📋 Problem Set

### 🧱 Fundamentals — "Implement X from scratch"

Common basics you'll encounter in interviews. Implement these without using `torch.nn`.

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 1 | ReLU | `relu(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | Activation functions, element-wise ops |
| 2 | Softmax | `my_softmax(x, dim)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | Numerical stability, exp/log tricks |
| 3 | Linear Layer | `SimpleLinear` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | `y = xW^T + b`, Kaiming init, `nn.Parameter` |
| 4 | LayerNorm | `my_layer_norm(x, γ, β)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | Normalization, running stats, affine transform |
| 7 | BatchNorm | `my_batch_norm(x, γ, β)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | Batch vs layer statistics, train/eval behavior |
| 8 | RMSNorm | `rms_norm(x, weight)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | LLaMA-style norm, simpler than LayerNorm |

### 🧠 Attention Mechanisms — The heart of modern ML

Helpful if you're preparing for LLM or Transformer related roles.

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 5 | Scaled Dot-Product Attention | `scaled_dot_product_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | `softmax(QK^T/√d_k)V`, the foundation of everything |
| 6 | Multi-Head Attention | `MultiHeadAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Parallel heads, split/concat, projection matrices |
| 9 | Causal Self-Attention | `causal_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Autoregressive masking with `-inf`, GPT-style |
| 10 | Grouped Query Attention | `GroupQueryAttention` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | GQA (LLaMA 2), KV sharing across heads |
| 11 | Sliding Window Attention | `sliding_window_attention(Q, K, V, w)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Mistral-style local attention, O(n·w) complexity |
| 12 | Linear Attention | `linear_attention(Q, K, V)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Kernel trick, `φ(Q)(φ(K)^TV)`, O(n·d²) |

### 🏗️ Full Architecture — Putting it all together

Combine what you've learned to implement complete components.

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 13 | GPT-2 Block | `GPT2Block` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Pre-norm, causal MHA + MLP (4x, GELU), residual connections |

### 🧪 Modern Activation Functions — LLM & Diffusion *(New in V2)*

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 14 | GELU | `gelu(x)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | Gaussian CDF, erf, used in BERT/GPT/DiT |
| 15 | SiLU (Swish) | `silu(x)` | ![Easy](https://img.shields.io/badge/Easy-4CAF50?style=flat-square) | x·σ(x), non-monotonic, LLaMA component |
| 16 | SwiGLU | `SwiGLU` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Gated activation, SiLU gate, LLaMA MLP |

### 🔧 Parameter-Efficient Fine-Tuning *(New in V2)*

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 17 | LoRA | `LoRALinear` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Low-rank BA, zero-init B, α/r scaling |
| 18 | DoRA | `DoRALinear` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Weight decomposition, magnitude + direction |

### 🎛️ Conditional Modulation — Diffusion & Style Transfer *(New in V2)*

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 19 | AdaLN | `AdaLN` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Adaptive LayerNorm, γ(c), β(c), DiT-style |
| 20 | AdaLN-Zero | `AdaLNZero` (nn.Module) | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Zero-init, gate for residual, stable training |
| 21 | FiLM | `FiLM` (nn.Module) | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | Feature-wise modulation, γ(c)·x + β(c) |

### 🚀 LLM Inference Components *(New in V2)*

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 22 | RoPE | `apply_rotary_pos_emb(x, pos)` | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Rotary embedding, LLaMA/Mistral, 2D rotation |
| 23 | KV Cache | `KVCache` class | ![Hard](https://img.shields.io/badge/Hard-F44336?style=flat-square) | Incremental caching, append/get, generation |

### 📊 Diffusion Training *(New in V2)*

| # | Problem | What You'll Implement | Difficulty | Key Concepts |
|:---:|---------|----------------------|:----------:|--------------|
| 24 | Sigmoid Schedule | `sigmoid_schedule(t, ...)` | ![Medium](https://img.shields.io/badge/Medium-FF9800?style=flat-square) | S-curve β schedule, smooth transition |

---

## ⚙️ How It Works

Each problem has **two** notebooks:

| File | Purpose |
|------|---------|
| `01_relu.ipynb` | ✏️ Blank template — write your code here |
| `01_relu_solution.ipynb` | 📖 Reference solution — check when stuck |

### Workflow

```text
1. Open a blank notebook           →  Read the problem description
2. Implement your solution         →  Use only basic PyTorch ops
3. Debug freely                    →  print(x.shape), check gradients, etc.
4. Run the judge cell              →  check("relu")
5. See instant colored feedback    →  ✅ pass / ❌ fail per test case
6. Stuck? Get a nudge              →  hint("relu")
7. Review the reference solution   →  01_relu_solution.ipynb
```

### In-Notebook API

```python
from torch_judge import check, hint, status

check("relu")               # Judge your implementation
hint("causal_attention")    # Get a hint without full spoiler
status()                    # Progress dashboard — solved / attempted / todo
```

---

## 🏛️ Project Structure

```
HappyTorch/
├── torch_judge/           # Auto-grading engine
│   ├── __init__.py        # Public API: check(), hint(), status()
│   ├── engine.py          # Core logic — extracts user function from IPython namespace
│   ├── progress.py        # Tracks solved/attempted status in data/progress.json
│   └── tasks/             # Task definitions (auto-discovered)
│       ├── _registry.py   # Auto-imports all TASK dicts from sibling modules
│       ├── relu.py        # Example task definition
│       ├── gelu.py        # ✨ New in V2
│       ├── lora.py        # ✨ New in V2
│       └── ...            # 24 tasks total
│
├── web/                   # 🌐 Web interface (FastAPI + Monaco Editor)
│   ├── app.py             # FastAPI backend, REST API for tasks/submit/progress
│   ├── requirements.txt   # Web-specific dependencies
│   └── static/
│       └── index.html     # Single-page app with Monaco code editor
│
├── templates/             # Blank notebooks for practice
├── solutions/             # Reference implementations
├── notebooks/             # User workspace (create this directory)
├── data/                  # Progress tracking (auto-created)
│
├── start_web.py           # Launch web server
├── start_jupyter.py       # Launch JupyterLab
├── setup.py               # Package installation
├── CODEBUDDY.md           # Guidance for AI coding assistants
└── README.md              # This file
```

---

## 📅 Suggested Study Plan

> **Total: ~10–12 hours spread across 3–4 weeks.**

| Week | Focus | Problems | Time |
|:----:|-------|----------|:----:|
| **1** | 🧱 Foundations | ReLU → Softmax → Linear → LayerNorm → BatchNorm → RMSNorm | 1–2 hrs |
| **2** | 🧠 Attention Deep Dive | SDPA → MHA → Causal → GQA → Sliding Window → Linear Attn | 3–4 hrs |
| **3** | 🧪 Modern Components | GELU → SiLU → SwiGLU → LoRA → DoRA | 2–3 hrs |
| **4** | 🎛️ Advanced Topics | AdaLN → FiLM → RoPE → KV Cache → GPT-2 Block | 3–4 hrs |

---

## 🧩 Adding Your Own Problems

HappyTorch uses auto-discovery — just drop a new file in `torch_judge/tasks/`:

```python
# torch_judge/tasks/my_task.py
TASK = {
    "title": "My Custom Problem",
    "difficulty": "Medium",  # Easy/Medium/Hard
    "function_name": "my_function",
    "hint": "Think about broadcasting...",
    "tests": [
        {"name": "Basic test", "code": "assert ..."},
        # ... more tests
    ]
}
```

No registration needed. The judge picks it up automatically. Then create corresponding notebooks in `templates/` and `solutions/`.

---

## ❓ FAQ

<details>
<summary><b>Do I need a GPU?</b></summary>
<br>
No. Everything runs on CPU. The problems test correctness and understanding, not throughput.
</details>

<details>
<summary><b>How are solutions graded?</b></summary>
<br>
The judge runs your function against multiple test cases using <code>torch.allclose</code> for numerical correctness, verifies gradients flow properly via autograd, and checks edge cases specific to each operation.
</details>

<details>
<summary><b>Can I save my progress?</b></summary>
<br>
Progress is saved in <code>data/progress.json</code>. Your solutions in <code>notebooks/</code> persist between sessions. To start fresh, simply re-copy templates.
</details>

<details>
<summary><b>What's different from the original TorchCode?</b></summary>
<br>
HappyTorch adds 11 new problems covering:
- Modern activation functions (GELU, SiLU, SwiGLU)
- Parameter-efficient fine-tuning (LoRA, DoRA)
- Diffusion model components (AdaLN, AdaLN-Zero, FiLM)
- LLM inference optimization (RoPE, KV Cache)
- Diffusion training (Sigmoid Schedule)
</details>

<details>
<summary><b>Who is this for?</b></summary>
<br>

This project is primarily for:

- Beginners learning deep learning who want to solidify their fundamentals
- Job seekers preparing for ML/AI related interviews
- Developers who want to understand PyTorch internals
- Practitioners interested in technologies like LLM, Diffusion, PEFT

Whether you're just starting out or have some experience, you can find practice problems suited to your level.
</details>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The original TorchCode project is also licensed under MIT.

---

<div align="center">

**Learning is a long journey. Hope this little project can help you along the way.**

If you find it useful, a Star ⭐ would be appreciated.

**Special thanks to [TorchCode](https://github.com/duoan/TorchCode) for the original foundation.**

</div>
