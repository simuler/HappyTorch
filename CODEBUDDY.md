# CODEBUDDY.md This file provides guidance to CodeBuddy when working with code in this repository.

## Commands

```bash
# Setup (first time)
conda create -n torchcode python=3.11 -y
conda activate torchcode
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install jupyterlab numpy
pip install -e .

# Install web dependencies (for web mode)
pip install fastapi uvicorn python-multipart

# Prepare notebooks (for Jupyter mode)
mkdir -p notebooks && cp -r templates/* notebooks/ && cp -r solutions/* notebooks/

# Start Web Mode (LeetCode-like interface)
python start_web.py
# Open http://localhost:8000

# Start Jupyter Mode (traditional notebooks)
python start_jupyter.py
# OR: jupyter lab --notebook-dir=notebooks --NotebookApp.token=''
```

Python 3.10+ required. PyTorch CPU-only, no GPU needed.

## Architecture

TorchCodeV2 is a PyTorch interview practice platform (LeetCode for tensors). Jupyter-based, no GPU required.

Based on [TorchCode](https://github.com/duoan/TorchCode) with 11 additional problems for LLM, Diffusion, and PEFT.

### Core Components

```
torch_judge/           # Auto-grading engine (pip install -e .)
├── __init__.py        # Public API: check(), hint(), status()
├── engine.py          # Core logic — extracts user function from IPython namespace
├── progress.py        # Tracks solved/attempted status in data/progress.json
└── tasks/             # Task definitions (auto-discovered)
    ├── _registry.py   # Auto-imports all TASK dicts from sibling modules
    └── *.py           # 24 tasks total (see categories below)

web/                   # Web interface (FastAPI + Monaco Editor)
├── app.py             # FastAPI backend, REST API for tasks/submit/progress
├── requirements.txt   # Web-specific dependencies
└── static/
    └── index.html     # Single-page app with Monaco code editor
```

### Workflow

1. User opens blank notebook from `templates/`
2. Implements a function (e.g., `relu(x)`)
3. Calls `check("relu")` — engine extracts function from `get_ipython().user_ns`
4. Tests run via `exec()` with assertions
5. Results shown with colored output; progress saved

### Task Definition Format

```python
# torch_judge/tasks/my_task.py
TASK = {
    "title": "Implement X",
    "difficulty": "Medium",  # Easy/Medium/Hard
    "function_name": "my_function",
    "hint": "Think about...",
    "tests": [
        {"name": "Basic test", "code": "assert ..."},
        ...
    ]
}
```

Tasks are auto-registered by `_registry.py` scanning the `tasks/` directory.

### Directory Structure

- `templates/` — Blank notebooks for practice
- `solutions/` — Reference implementations
- `notebooks/` — User workspace (create this directory)
- `data/` — Progress tracking (auto-created)

### Task Categories

**Fundamentals (Easy/Medium):**
- `relu`, `softmax`, `silu`, `gelu` — Activation functions

**Normalization & Layers (Medium):**
- `linear`, `layernorm`, `batchnorm`, `rmsnorm`

**Attention Mechanisms (Hard):**
- `attention`, `mha`, `causal_attention`, `gqa`, `sliding_window`, `linear_attention`

**Modern LLM Components (Hard):**
- `swiglu` — SwiGLU activation (LLaMA-style)
- `rope` — Rotary Position Embedding
- `kv_cache` — KV Cache for inference

**Parameter-Efficient Fine-Tuning (Hard):**
- `lora` — Low-Rank Adaptation
- `dora` — Weight-Decomposed LoRA

**Conditional Modulation (Medium/Hard):**
- `adaln` — Adaptive Layer Normalization (DiT-style)
- `adaln_zero` — Zero-initialized AdaLN
- `film` — Feature-wise Linear Modulation

**Diffusion Training (Medium):**
- `snr` — Sigmoid Noise Schedule

**Full Architecture (Hard):**
- `gpt2_block` — Complete transformer block

### Key Implementation Details

- `engine._get_user_namespace()` uses IPython's `get_ipython().user_ns` to access notebook variables
- Tests are executed via `exec()` with `{fn}` placeholder replaced by actual function name
- Progress stored in `data/progress.json` with status, best_time, attempts
