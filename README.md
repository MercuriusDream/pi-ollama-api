# pi-ollama-api

Ollama Cloud provider extension for [Pi](https://pi.dev) — connect your terminal coding agent to **200+ models** on Ollama Cloud via the OpenAI-compatible API.

## Features

- **Dynamic model discovery** — Fetches live model list from `ollama.com/v1/models` on startup
- **194 known model metadata entries** — Context windows, vision support, reasoning flags, and descriptions for all major Ollama families
- **OpenAI-compatible API** — Uses `openai-completions` streaming (works with all Pi features)
- **Embeddings tool** — Generate embeddings via `/v1/embeddings` for RAG and similarity search
- **Direct chat tool** — Send one-off completions for model comparison or testing
- **Vision support** — Correctly flags vision-capable models (Llama 3.2 Vision, Gemma 3, Qwen VL, etc.)
- **Reasoning support** — Correctly flags reasoning models (DeepSeek R1, Qwen QWQ, etc.)

## Supported Model Families

| Family | Models | Highlights |
|--------|--------|------------|
| **Llama** | 3.3, 3.2, 3.1, 3, 2 | 70B frontier, Vision variants, 405B |
| **Qwen** | 3, 2.5, 2, VL, Coder, Math | 128K context, Vision, Code, Math variants |
| **DeepSeek** | R1, V3, V2, Coder V2 | Reasoning (R1), 671B total |
| **Mistral** | Codestral, Mistral, Nemo, Large, Mixtral | 256K context Codestral |
| **Gemma** | 3, 2, CodeGemma, ShieldGemma | Vision support, 128K context |
| **Phi** | 4, 3.5, 3 | Microsoft models, 128K context |
| **IBM** | Granite 3.x, Granite Code | MoE variants, 128K context |
| **Cohere** | Command R, Aya, Aya Expanse | Multilingual, 128K context |
| **GPT-OSS** | 120B, 20B (Cloud) | Cloud-hosted OSS models |
| **+ 30+ more** | Yi, Falcon, GLM, InternLM, SOLAR, etc. | See full list in source |

## Installation

```bash
# Install via pi
pi install npm:pi-ollama-api

# Or install locally
pi install npm:pi-ollama-api -l
```

## Setup

1. **Get an API key** from [ollama.com/settings](https://ollama.com/settings)
2. **Start Pi** and run:
   ```
   /ollama-cloud-login
   ```
   Paste your API key when prompted. It is stored in Pi's `~/.pi/agent/auth.json` (same place as `/login` credentials).
3. **Select a model** with `/model` → pick any `ollama-cloud/*` model

## Authentication

| Method | How | Where stored |
|--------|-----|-------------|
| **Interactive (recommended)** | Run `/ollama-cloud-login` in Pi | `~/.pi/agent/auth.json` |
| **Environment variable** | `export OLLAMA_API_KEY=...` | Shell env |

Pi's `AuthStorage` is used natively — API keys are checked in `auth.json` first, then the env var is used as a fallback.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_API_KEY` | — | Fallback API key (used if auth.json has no key) |
| `OLLAMA_CLOUD_BASE_URL` | `https://ollama.com/v1` | Override endpoint (for proxies or self-hosted) |
| `OLLAMA_CLOUD_MODELS` | — | Comma-separated list to skip discovery and use static models |
| `OLLAMA_CLOUD_TIMEOUT` | `30000` | Model discovery timeout in ms |

## Usage

### Select a model

```
/model
```

Then pick any `ollama-cloud/*` model. Examples:
- `ollama-cloud/llama3.3` — Llama 3.3 70B
- `ollama-cloud/qwen3` — Qwen 3 with vision
- `ollama-cloud/deepseek-r1` — DeepSeek R1 with reasoning
- `ollama-cloud/gemma3:27b` — Gemma 3 27B with vision

### Commands

| Command | Description |
|---------|-------------|
| `/ollama-cloud-status` | Check API key status and model count |
| `/ollama-cloud-refresh` | Re-fetch live model list from Ollama Cloud API |
| `/ollama-cloud-list` | Pretty-print all models with 🧠/🖼️/💬 badges |
| `/ollama-cloud-pull <id>` | Show the `ollama pull` command for a model |

### Tools (LLM-callable)

| Tool | Purpose |
|------|---------|
| `ollama_list_models` | Filter models by family, vision, or reasoning |
| `ollama_embeddings` | Generate embeddings via `/v1/embeddings` |
| `ollama_chat` | Direct chat completion via `/v1/chat/completions` |
| `ollama_model_info` | Get detailed metadata for a specific model |

## Quick Examples

```
# Check what models are available
Use ollama_list_models to show all available models

# Get embeddings for a document
Use ollama_embeddings with model "nomic-embed-text" and input "The quick brown fox"

# Compare model outputs
Use ollama_chat with model "llama3.3" and messages [{role: "user", content: "Hello"}]
Use ollama_chat with model "qwen3" and messages [{role: "user", content: "Hello"}]
```

## API Compatibility

This extension uses Ollama's **OpenAI-compatible API** (`/v1/chat/completions`), which supports:
- Chat completions with streaming
- Vision (multimodal) inputs
- Tool calling
- JSON mode
- Reasoning/thinking control
- Embeddings (`/v1/embeddings`)

## License

MIT
