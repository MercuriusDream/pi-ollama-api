/**
 * Ollama Cloud Provider Extension for Pi
 *
 * Full-featured Ollama Cloud integration with OpenAI-compatible API.
 * Provides provider registration, model discovery, custom tools for
 * embeddings/model info, and slash commands for Ollama operations.
 *
 * Setup:
 *   1. Get API key from https://ollama.com/settings
 *   2. export OLLAMA_API_KEY=your-api-key
 *   3. Place this file in ~/.pi/agent/extensions/ollama-cloud.ts
 *   4. Run `/reload` in Pi or restart
 *   5. Use `/model` → select ollama-cloud/* models
 *
 * Commands:
 *   /ollama-cloud-status    — Show provider status
 *   /ollama-cloud-refresh   — Refresh model list from API
 *   /ollama-cloud-list      — Pretty-print available models
 *   /ollama-cloud-pull <id> — Pull a model from Ollama Hub
 *
 * Environment overrides:
 *   OLLAMA_CLOUD_BASE_URL — Custom endpoint (default: https://ollama.com/v1)
 *   OLLAMA_CLOUD_MODELS   — Comma-separated override list (skips discovery)
 *   OLLAMA_CLOUD_TIMEOUT  — Request timeout ms (default: 30000)
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { AuthStorage } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// =============================================================================
// Types
// =============================================================================

interface ModelMeta {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  contextWindow: number;
  maxTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
  family?: string;
  size?: string;
  description?: string;
}

interface OllamaModelEntry {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

// =============================================================================
// Comprehensive Model Metadata — All Known Ollama Cloud Models
// =============================================================================

const KNOWN_MODELS: Record<string, Partial<ModelMeta>> = {
  // ─── Llama Family ──────────────────────────────────────────────
  "llama3.3":               { name: "Llama 3.3 70B",               family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Meta's latest 70B general-purpose model" },
  "llama3.2":               { name: "Llama 3.2 3B",                family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Lightweight 3B edge model" },
  "llama3.2:1b":            { name: "Llama 3.2 1B",                family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Tiny 1B model for mobile/edge" },
  "llama3.2-vision":        { name: "Llama 3.2 Vision 11B",        family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Vision-enabled 11B model" },
  "llama3.2-vision:90b":    { name: "Llama 3.2 Vision 90B",        family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "High-capability vision 90B" },
  "llama3.1":               { name: "Llama 3.1 8B",               family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Balanced 8B general-purpose" },
  "llama3.1:70b":           { name: "Llama 3.1 70B",               family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Strong 70B reasoning model" },
  "llama3.1:405b":          { name: "Llama 3.1 405B",              family: "llama",      contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Massive 405B frontier model" },
  "llama3":                 { name: "Llama 3 8B",                  family: "llama",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Original Llama 3" },
  "llama3:70b":             { name: "Llama 3 70B",                  family: "llama",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Original Llama 3 70B" },
  "llama2":                 { name: "Llama 2 7B",                  family: "llama",      contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Legacy Llama 2" },
  "llama2:70b":             { name: "Llama 2 70B",                  family: "llama",      contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Legacy Llama 2 70B" },

  // ─── Qwen Family ───────────────────────────────────────────────
  "qwen3":                  { name: "Qwen 3",                     family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Alibaba's latest Qwen 3" },
  "qwen3:30b":              { name: "Qwen 3 30B",                  family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Mid-size Qwen 3 30B" },
  "qwen3:72b":              { name: "Qwen 3 72B",                  family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Large Qwen 3 72B" },
  "qwen2.5":                { name: "Qwen 2.5 7B",                 family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Fast 7B general model" },
  "qwen2.5:14b":            { name: "Qwen 2.5 14B",                family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Mid-size Qwen 2.5" },
  "qwen2.5:32b":            { name: "Qwen 2.5 32B",                family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Strong 32B Qwen 2.5" },
  "qwen2.5:72b":            { name: "Qwen 2.5 72B",                family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Large 72B Qwen 2.5" },
  "qwen2.5-coder":          { name: "Qwen 2.5 Coder 14B",          family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Code-specialized 14B" },
  "qwen2.5-coder:32b":      { name: "Qwen 2.5 Coder 32B",          family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Code-specialized 32B" },
  "qwen2.5-math":           { name: "Qwen 2.5 Math 7B",            family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Math-specialized 7B" },
  "qwen2.5-math:72b":       { name: "Qwen 2.5 Math 72B",           family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Math-specialized 72B" },
  "qwen2":                  { name: "Qwen 2 7B",                  family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Legacy Qwen 2" },
  "qwen2:72b":              { name: "Qwen 2 72B",                  family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Legacy Qwen 2 72B" },
  "qwen2-vl":               { name: "Qwen 2 VL 7B",                family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text", "image"],  description: "Vision-language Qwen 2" },
  "qwen2-vl:72b":           { name: "Qwen 2 VL 72B",               family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text", "image"],  description: "Vision-language Qwen 2 72B" },
  "qwen2.5-vl":             { name: "Qwen 2.5 VL 7B",              family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Vision-language Qwen 2.5" },
  "qwen2.5-vl:72b":         { name: "Qwen 2.5 VL 72B",             family: "qwen",       contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Vision-language Qwen 2.5 72B" },

  // ─── Mistral / Codestral / Mixtral ─────────────────────────────
  "codestral":              { name: "Codestral 22B",               family: "mistral",    contextWindow: 256000, maxTokens: 8192,  input: ["text"],           description: "Code-specialized 22B" },
  "codestral:2505":         { name: "Codestral 2505 22B",          family: "mistral",    contextWindow: 256000, maxTokens: 8192,  input: ["text"],           description: "Codestral 2505 update" },
  "mistral":                { name: "Mistral 7B",                  family: "mistral",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Fast 7B general model" },
  "mistral-nemo":           { name: "Mistral Nemo 12B",            family: "mistral",    contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Mistral Nemo 12B" },
  "mistral-large":          { name: "Mistral Large 123B",          family: "mistral",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Frontier Mistral Large" },
  "mistral-small":          { name: "Mistral Small 22B",          family: "mistral",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Efficient Mistral Small" },
  "mixtral":                { name: "Mixtral 8x7B",                family: "mistral",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Sparse MoE 8x7B" },
  "mixtral:8x22b":          { name: "Mixtral 8x22B",               family: "mistral",    contextWindow: 65536,  maxTokens: 4096,  input: ["text"],           description: "Sparse MoE 8x22B" },
  "mathstral":              { name: "Mathstral 7B",                family: "mistral",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Math-specialized Mistral" },

  // ─── DeepSeek ───────────────────────────────────────────────────
  "deepseek-coder-v2":      { name: "DeepSeek Coder V2 16B",      family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "MoE code model 16B active" },
  "deepseek-coder-v2:236b": { name: "DeepSeek Coder V2 236B",     family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "MoE code model 236B total" },
  "deepseek-v3":            { name: "DeepSeek V3",                family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "DeepSeek V3 general MoE" },
  "deepseek-r1":            { name: "DeepSeek R1 7B",              family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           reasoning: true, description: "Reasoning model 7B active" },
  "deepseek-r1:14b":        { name: "DeepSeek R1 14B",             family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           reasoning: true, description: "Reasoning model 14B active" },
  "deepseek-r1:32b":        { name: "DeepSeek R1 32B",             family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           reasoning: true, description: "Reasoning model 32B active" },
  "deepseek-r1:70b":        { name: "DeepSeek R1 70B",             family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           reasoning: true, description: "Reasoning model 70B active" },
  "deepseek-r1:671b":       { name: "DeepSeek R1 671B",            family: "deepseek",   contextWindow: 128000, maxTokens: 8192,  input: ["text"],           reasoning: true, description: "Full MoE reasoning 671B total" },
  "deepseek-v2":            { name: "DeepSeek V2 16B",             family: "deepseek",   contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "DeepSeek V2 general MoE" },
  "deepseek-v2:236b":       { name: "DeepSeek V2 236B",            family: "deepseek",   contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "DeepSeek V2 236B total" },
  "deepseek-llm":           { name: "DeepSeek LLM 67B",            family: "deepseek",   contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "DeepSeek base LLM 67B" },

  // ─── Google Gemma ───────────────────────────────────────────────
  "gemma3":                 { name: "Gemma 3 4B",                  family: "gemma",      contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Google's lightweight 4B with vision" },
  "gemma3:1b":              { name: "Gemma 3 1B",                  family: "gemma",      contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Tiny 1B vision model" },
  "gemma3:12b":             { name: "Gemma 3 12B",                 family: "gemma",      contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Mid-size 12B vision model" },
  "gemma3:27b":             { name: "Gemma 3 27B",                 family: "gemma",      contextWindow: 128000, maxTokens: 8192,  input: ["text", "image"],  description: "Strong 27B vision model" },
  "gemma2":                 { name: "Gemma 2 9B",                  family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Gemma 2 9B" },
  "gemma2:27b":             { name: "Gemma 2 27B",                 family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Gemma 2 27B" },
  "gemma":                  { name: "Gemma 2B",                    family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Original Gemma 2B" },
  "gemma:7b":               { name: "Gemma 7B",                    family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Original Gemma 7B" },

  // ─── Microsoft Phi ────────────────────────────────────────────────
  "phi4":                   { name: "Phi 4 14B",                   family: "phi",        contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Microsoft Phi 4 14B" },
  "phi4-mini":              { name: "Phi 4 Mini 3.8B",             family: "phi",        contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Compact Phi 4 Mini" },
  "phi3":                   { name: "Phi 3 3.8B",                  family: "phi",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Microsoft Phi 3" },
  "phi3:14b":               { name: "Phi 3 14B",                   family: "phi",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Phi 3 medium 14B" },
  "phi3:medium":            { name: "Phi 3 Medium 14B",            family: "phi",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Phi 3 medium 14B" },
  "phi3.5":                 { name: "Phi 3.5 3.8B",                family: "phi",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Phi 3.5 mini" },
  "phi3.5-vision":          { name: "Phi 3.5 Vision 4.2B",         family: "phi",        contextWindow: 128000, maxTokens: 4096,  input: ["text", "image"],  description: "Phi 3.5 vision model" },

  // ─── GPT-OSS (Cloud-only) ──────────────────────────────────────
  "gpt-oss":                { name: "GPT-OSS",                     family: "gpt-oss",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "OpenAI-style OSS model" },
  "gpt-oss:20b":            { name: "GPT-OSS 20B",                 family: "gpt-oss",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "GPT-OSS 20B parameter" },
  "gpt-oss:120b":           { name: "GPT-OSS 120B",                family: "gpt-oss",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "GPT-OSS 120B parameter" },
  "gpt-oss:120b-cloud":     { name: "GPT-OSS 120B (Cloud)",      family: "gpt-oss",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Cloud-hosted GPT-OSS 120B" },
  "gpt-oss:20b-cloud":      { name: "GPT-OSS 20B (Cloud)",         family: "gpt-oss",    contextWindow: 128000, maxTokens: 8192,  input: ["text"],           description: "Cloud-hosted GPT-OSS 20B" },

  // ─── Cohere / Command ───────────────────────────────────────────
  "command-r":              { name: "Command R 35B",               family: "cohere",     contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Cohere Command R 35B" },
  "command-r-plus":         { name: "Command R+ 104B",            family: "cohere",     contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Cohere Command R+ 104B" },
  "command-r7b":            { name: "Command R7B 7B",              family: "cohere",     contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Cohere Command R7B" },
  "command-r7b:12b":       { name: "Command R7B 12B",             family: "cohere",     contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Cohere Command R7B 12B" },
  "aya":                    { name: "Aya 8B",                      family: "cohere",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Cohere Aya 8B multilingual" },
  "aya:35b":                { name: "Aya 35B",                     family: "cohere",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Cohere Aya 35B multilingual" },
  "aya-expanse":            { name: "Aya Expanse 8B",              family: "cohere",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Cohere Aya Expanse 8B" },
  "aya-expanse:32b":        { name: "Aya Expanse 32B",             family: "cohere",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Cohere Aya Expanse 32B" },
  "aya-vision":             { name: "Aya Vision 8B",               family: "cohere",     contextWindow: 8192,   maxTokens: 4096,  input: ["text", "image"],  description: "Cohere Aya Vision 8B" },
  "aya-vision:32b":         { name: "Aya Vision 32B",              family: "cohere",     contextWindow: 8192,   maxTokens: 4096,  input: ["text", "image"],  description: "Cohere Aya Vision 32B" },

  // ─── Nous / Hermes / WizardLM / Yi / etc ────────────────────────
  "nous-hermes2":           { name: "Nous Hermes 2 11B",           family: "nous",       contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Nous Hermes 2" },
  "nous-hermes2:34b":       { name: "Nous Hermes 2 34B",           family: "nous",       contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Nous Hermes 2 34B" },
  "nous-hermes2-mixtral":   { name: "Nous Hermes 2 Mixtral",       family: "nous",       contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Nous Hermes 2 Mixtral" },
  "wizardlm2":              { name: "WizardLM 2 7B",               family: "wizard",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "WizardLM 2 7B" },
  "wizardlm2:8x22b":        { name: "WizardLM 2 8x22B",            family: "wizard",     contextWindow: 65536,  maxTokens: 4096,  input: ["text"],           description: "WizardLM 2 Mixtral 8x22B" },
  "yi":                     { name: "Yi 6B",                       family: "yi",         contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "01.AI Yi 6B" },
  "yi:34b":                 { name: "Yi 34B",                      family: "yi",         contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "01.AI Yi 34B" },
  "dolphin-llama3":         { name: "Dolphin Llama 3 8B",          family: "dolphin",    contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Dolphin uncensored Llama 3" },
  "dolphin-llama3:70b":     { name: "Dolphin Llama 3 70B",         family: "dolphin",    contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Dolphin uncensored Llama 3 70B" },
  "dolphin-mixtral":        { name: "Dolphin Mixtral 8x7B",        family: "dolphin",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Dolphin uncensored Mixtral" },
  "dolphin-mistral":        { name: "Dolphin Mistral 7B",            family: "dolphin",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Dolphin uncensored Mistral" },
  "orca-mini":              { name: "Orca Mini 3B",                family: "orca",       contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Microsoft Orca Mini 3B" },
  "orca-mini:7b":           { name: "Orca Mini 7B",                family: "orca",       contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Microsoft Orca Mini 7B" },
  "orca-mini:13b":          { name: "Orca Mini 13B",               family: "orca",       contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Microsoft Orca Mini 13B" },
  "orca-mini:70b":          { name: "Orca Mini 70B",               family: "orca",       contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Microsoft Orca Mini 70B" },
  "starling-lm":            { name: "Starling LM 7B",              family: "starling",   contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Berkeley Starling LM" },
  "neural-chat":            { name: "Neural Chat 7B",              family: "intel",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Intel Neural Chat" },
  "neural-chat:7b-v3-3":    { name: "Neural Chat 7B v3.3",         family: "intel",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Intel Neural Chat v3.3" },
  "openchat":               { name: "OpenChat 3.5 7B",             family: "openchat",   contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "OpenChat 3.5" },
  "solar":                  { name: "SOLAR 10.7B",                 family: "solar",      contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Upstage SOLAR" },
  "solar-pro":              { name: "SOLAR Pro 22B",                 family: "solar",      contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Upstage SOLAR Pro" },
  "falcon":                 { name: "Falcon 7B",                   family: "falcon",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "TII Falcon 7B" },
  "falcon:40b":             { name: "Falcon 40B",                  family: "falcon",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "TII Falcon 40B" },
  "falcon2":                { name: "Falcon 2 11B",                family: "falcon",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "TII Falcon 2 11B" },
  "granite3-dense":         { name: "Granite 3 Dense 8B",         family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3 Dense" },
  "granite3-dense:2b":      { name: "Granite 3 Dense 2B",          family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3 Dense 2B" },
  "granite3-moe":           { name: "Granite 3 MoE 3B",            family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3 MoE" },
  "granite3-moe:1b":        { name: "Granite 3 MoE 1B",            family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3 MoE 1B" },
  "granite3.1-dense":       { name: "Granite 3.1 Dense 8B",        family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3.1 Dense" },
  "granite3.1-moe":         { name: "Granite 3.1 MoE 3B",          family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3.1 MoE" },
  "granite-code":           { name: "Granite Code 8B",              family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite Code 8B" },
  "granite-code:20b":       { name: "Granite Code 20B",            family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite Code 20B" },
  "starcoder2":             { name: "StarCoder2 3B",               family: "bigcode",    contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "BigCode StarCoder2" },
  "starcoder2:7b":          { name: "StarCoder2 7B",               family: "bigcode",    contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "BigCode StarCoder2 7B" },
  "starcoder2:15b":         { name: "StarCoder2 15B",              family: "bigcode",    contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "BigCode StarCoder2 15B" },
  "codegemma":              { name: "CodeGemma 2B",                  family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google CodeGemma 2B" },
  "codegemma:7b":           { name: "CodeGemma 7B",                  family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google CodeGemma 7B" },
  "codegemma:code":         { name: "CodeGemma Code 7B",           family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google CodeGemma Code 7B" },
  "codellama":              { name: "CodeLlama 7B",                family: "llama",      contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "Meta CodeLlama 7B" },
  "codellama:13b":          { name: "CodeLlama 13B",               family: "llama",      contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "Meta CodeLlama 13B" },
  "codellama:34b":          { name: "CodeLlama 34B",               family: "llama",      contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "Meta CodeLlama 34B" },
  "codellama:70b":          { name: "CodeLlama 70B",               family: "llama",      contextWindow: 16384,  maxTokens: 4096,  input: ["text"],           description: "Meta CodeLlama 70B" },
  "tinyllama":              { name: "TinyLlama 1.1B",              family: "llama",      contextWindow: 2048,   maxTokens: 2048,  input: ["text"],           description: "TinyLlama 1.1B chat" },
  "tinydolphin":            { name: "TinyDolphin 2.8B",            family: "dolphin",    contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "TinyDolphin 2.8B" },
  "stablelm2":              { name: "StableLM 2 1.6B",             family: "stable",     contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Stability AI StableLM 2" },
  "stablelm2:12b":          { name: "StableLM 2 12B",              family: "stable",     contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Stability AI StableLM 2 12B" },
  "stablelm2:zephyr":       { name: "StableLM Zephyr 3B",          family: "stable",     contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Stability AI StableLM Zephyr" },
  "internlm2":              { name: "InternLM 2 7B",               family: "internlm",   contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Shanghai AI InternLM 2" },
  "internlm2:20b":          { name: "InternLM 2 20B",              family: "internlm",   contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Shanghai AI InternLM 2 20B" },
  "baichuan2":              { name: "Baichuan 2 7B",               family: "baichuan",   contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Baichuan 2 7B" },
  "baichuan2:13b":          { name: "Baichuan 2 13B",              family: "baichuan",   contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Baichuan 2 13B" },
  "llava":                  { name: "LLaVA 7B",                    family: "llava",      contextWindow: 4096,   maxTokens: 4096,  input: ["text", "image"],  description: "Vision-language LLaVA" },
  "llava:13b":              { name: "LLaVA 13B",                    family: "llava",      contextWindow: 4096,   maxTokens: 4096,  input: ["text", "image"],  description: "Vision-language LLaVA 13B" },
  "llava:34b":              { name: "LLaVA 34B",                    family: "llava",      contextWindow: 4096,   maxTokens: 4096,  input: ["text", "image"],  description: "Vision-language LLaVA 34B" },
  "llava-phi3":             { name: "LLaVA-Phi 3 3.8B",            family: "llava",      contextWindow: 4096,   maxTokens: 4096,  input: ["text", "image"],  description: "Lightweight LLaVA on Phi 3" },
  "llava-llama3":           { name: "LLaVA-Llama 3 8B",            family: "llava",      contextWindow: 4096,   maxTokens: 4096,  input: ["text", "image"],  description: "LLaVA on Llama 3" },
  "bakllava":               { name: "BakLLaVA 7B",                 family: "llava",      contextWindow: 4096,   maxTokens: 4096,  input: ["text", "image"],  description: "BakLLaVA vision model" },
  "moondream":              { name: "Moondream 1.6B",              family: "moondream",  contextWindow: 2048,   maxTokens: 2048,  input: ["text", "image"],  description: "Tiny vision model Moondream" },
  "moondream2":             { name: "Moondream 2 1.6B",            family: "moondream",  contextWindow: 2048,   maxTokens: 2048,  input: ["text", "image"],  description: "Moondream 2 vision model" },
  "minicpm-v":              { name: "MiniCPM-V 2.6 8B",            family: "minicpm",    contextWindow: 128000, maxTokens: 4096,  input: ["text", "image"],  description: "MiniCPM-V vision model" },
  "nomic-embed-text":       { name: "Nomic Embed Text",            family: "nomic",      contextWindow: 8192,   maxTokens: 8192,  input: ["text"],           description: "Nomic text embedding model" },
  "mxbai-embed-large":      { name: "mxbai-embed-large",           family: "mixedbread", contextWindow: 512,    maxTokens: 512,   input: ["text"],           description: "Mixedbread large embeddings" },
  "snowflake-arctic-embed": { name: "Snowflake Arctic Embed",      family: "snowflake",  contextWindow: 512,    maxTokens: 512,   input: ["text"],           description: "Snowflake embedding model" },
  "snowflake-arctic-embed:335m": { name: "Snowflake Arctic Embed 335M", family: "snowflake", contextWindow: 512, maxTokens: 512, input: ["text"], description: "Snowflake Arctic Embed 335M" },
  "snowflake-arctic-embed:l": { name: "Snowflake Arctic Embed L",  family: "snowflake",  contextWindow: 512,    maxTokens: 512,   input: ["text"],           description: "Snowflake Arctic Embed Large" },
  "bge-m3":                 { name: "BGE-M3",                      family: "bge",        contextWindow: 8192,   maxTokens: 8192,  input: ["text"],           description: "BAAI BGE-M3 embedding" },
  "bge-large":              { name: "BGE Large",                   family: "bge",        contextWindow: 512,    maxTokens: 512,   input: ["text"],           description: "BAAI BGE large embedding" },
  "all-minilm":             { name: "all-minilm",                  family: "sentence",   contextWindow: 512,    maxTokens: 512,   input: ["text"],           description: "Sentence Transformers MiniLM" },
  "smollm":                 { name: "SmolLM 1.7B",                 family: "smollm",     contextWindow: 2048,   maxTokens: 2048,  input: ["text"],           description: "Hugging Face SmolLM" },
  "smollm:360m":            { name: "SmolLM 360M",                 family: "smollm",     contextWindow: 2048,   maxTokens: 2048,  input: ["text"],           description: "Hugging Face SmolLM 360M" },
  "smollm2":                { name: "SmolLM2 1.7B",                family: "smollm",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Hugging Face SmolLM2" },
  "smollm2:360m":           { name: "SmolLM2 360M",                family: "smollm",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Hugging Face SmolLM2 360M" },
  "smollm2:1.7b":           { name: "SmolLM2 1.7B",                family: "smollm",     contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Hugging Face SmolLM2 1.7B" },
  "reader-lm":              { name: "ReaderLM 1.5B",               family: "reader",     contextWindow: 256000, maxTokens: 4096,  input: ["text"],           description: "Jina ReaderLM HTML-to-markdown" },
  "reader-lm:2b":           { name: "ReaderLM 2B",                 family: "reader",     contextWindow: 256000, maxTokens: 4096,  input: ["text"],           description: "Jina ReaderLM 2B" },
  "hermes3":                { name: "Hermes 3 8B",                 family: "nous",       contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Nous Hermes 3" },
  "hermes3:70b":            { name: "Hermes 3 70B",                family: "nous",       contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Nous Hermes 3 70B" },
  "athene-v2":              { name: "Athene V2 72B",               family: "nous",       contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "Athene V2 72B" },
  "mathstral:7b":           { name: "Mathstral 7B",                family: "mistral",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Mistral Mathstral 7B" },
  "mathstral:25b":          { name: "Mathstral 25B",               family: "mistral",    contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Mistral Mathstral 25B" },
  "qwq":                    { name: "QWQ 32B",                     family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           reasoning: true, description: "Qwen QWQ reasoning 32B" },
  "qwq:32b":                { name: "QWQ 32B Preview",             family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           reasoning: true, description: "Qwen QWQ 32B preview" },
  "marco-o1":               { name: "Marco-o1 7B",                 family: "qwen",       contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           reasoning: true, description: "Alibaba Marco-o1 reasoning" },
  "exaone3.5":              { name: "EXAONE 3.5 2.4B",             family: "lg",         contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "LG EXAONE 3.5" },
  "exaone3.5:7.8b":         { name: "EXAONE 3.5 7.8B",             family: "lg",         contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "LG EXAONE 3.5 7.8B" },
  "exaone3.5:32b":          { name: "EXAONE 3.5 32B",              family: "lg",         contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "LG EXAONE 3.5 32B" },
  "nemotron":               { name: "Nemotron 4 340B",             family: "nvidia",     contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "NVIDIA Nemotron 4" },
  "nemotron-mini":          { name: "Nemotron Mini 4B",            family: "nvidia",     contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "NVIDIA Nemotron Mini" },
  "nemotron:70b":           { name: "Nemotron 70B",              family: "nvidia",     contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "NVIDIA Nemotron 70B" },
  "samantha-mistral":       { name: "Samantha Mistral 7B",         family: "samantha",   contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Samantha companion Mistral" },
  "samantha-mistral:70b":   { name: "Samantha Mistral 70B",        family: "samantha",   contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Samantha companion Mistral 70B" },
  "sailor2":                { name: "Sailor2 8B",                  family: "sailor",     contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Sailor2 maritime LLM" },
  "sailor2:20b":            { name: "Sailor2 20B",                 family: "sailor",     contextWindow: 32768,  maxTokens: 4096,  input: ["text"],           description: "Sailor2 maritime LLM 20B" },
  "glm4":                   { name: "GLM-4 9B",                    family: "glm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "THUDM GLM-4 9B" },
  "glm4:32b":               { name: "GLM-4 32B",                   family: "glm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "THUDM GLM-4 32B" },
  "glm4:9b-chat":           { name: "GLM-4 9B Chat",               family: "glm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "THUDM GLM-4 9B chat" },
  "reflection":             { name: "Reflection 70B",              family: "reflection", contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "Reflection 70B self-correcting" },
  "opencoder":              { name: "OpenCoder 1.5B",               family: "opencoder",  contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "OpenCoder code model" },
  "opencoder:8b":           { name: "OpenCoder 8B",                family: "opencoder",  contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "OpenCoder 8B code model" },
  "opencoder:15b":          { name: "OpenCoder 15B",               family: "opencoder",  contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "OpenCoder 15B code model" },
  "opencoder:32b":          { name: "OpenCoder 32B",               family: "opencoder",  contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "OpenCoder 32B code model" },
  "llama-guard3":           { name: "Llama Guard 3 8B",            family: "llama",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Meta Llama Guard 3 safety" },
  "llama-guard3:1b":        { name: "Llama Guard 3 1B",            family: "llama",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Meta Llama Guard 3 1B" },
  "shieldgemma":            { name: "ShieldGemma 2B",              family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google ShieldGemma safety" },
  "shieldgemma:4b":         { name: "ShieldGemma 4B",              family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google ShieldGemma 4B" },
  "shieldgemma:9b":         { name: "ShieldGemma 9B",              family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google ShieldGemma 9B" },
  "shieldgemma:27b":        { name: "ShieldGemma 27B",             family: "gemma",      contextWindow: 8192,   maxTokens: 4096,  input: ["text"],           description: "Google ShieldGemma 27B" },
  "nuextract":              { name: "NuExtract 3.8B",              family: "nuextract",  contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "NuExtract structured extraction" },
  "nuextract:1.5b":         { name: "NuExtract 1.5B",              family: "nuextract",  contextWindow: 4096,   maxTokens: 4096,  input: ["text"],           description: "NuExtract structured extraction 1.5B" },
  "paraphrase-multilingual": { name: "Paraphrase Multilingual",     family: "sentence",   contextWindow: 512,    maxTokens: 512,   input: ["text"],           description: "Multilingual paraphrase embedding" },
  "lgranite3.2":            { name: "Granite 3.2 8B",               family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3.2 8B" },
  "lgranite3.2:2b":         { name: "Granite 3.2 2B",              family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3.2 2B" },
  "lgranite3.2:3b-audio":   { name: "Granite 3.2 3B Audio",        family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text"],           description: "IBM Granite 3.2 3B audio" },
  "lgranite3.2-vision":     { name: "Granite 3.2 Vision 2B",        family: "ibm",        contextWindow: 128000, maxTokens: 4096,  input: ["text", "image"],  description: "IBM Granite 3.2 Vision 2B" },
};

// Default metadata for unknown models
const DEFAULT_META: Partial<ModelMeta> = {
  contextWindow: 128000,
  maxTokens: 4096,
  input: ["text"],
  reasoning: false,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};

// =============================================================================
// Helpers
// =============================================================================

function resolveModelMeta(modelId: string): ModelMeta {
  const known = KNOWN_MODELS[modelId];
  return {
    id: modelId,
    name: known?.name || modelId,
    reasoning: known?.reasoning ?? false,
    input: (known?.input as ("text" | "image")[]) || ["text"],
    contextWindow: known?.contextWindow ?? DEFAULT_META.contextWindow!,
    maxTokens: known?.maxTokens ?? DEFAULT_META.maxTokens!,
    cost: known?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    family: known?.family,
    size: known?.size,
    description: known?.description,
  };
}

function makeHeaders(apiKey?: string): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

// ─── Provider Registration Helper ───────────────────────────────────────
function registerProviderWithKey(pi: ExtensionAPI, baseUrl: string, models: ModelMeta[]) {
  pi.registerProvider("ollama-cloud", {
    name: "Ollama Cloud",
    baseUrl,
    apiKey: "$OLLAMA_API_KEY",
    api: "openai-completions",
    authHeader: true,
    models: models.map((m) => ({
      id: m.id,
      name: m.name,
      reasoning: m.reasoning,
      input: m.input,
      cost: m.cost,
      contextWindow: m.contextWindow,
      maxTokens: m.maxTokens,
    })),
  });
}

// =============================================================================
// Dynamic Model Discovery
// =============================================================================

async function fetchOllamaModels(baseUrl: string, apiKey?: string): Promise<ModelMeta[]> {
  const response = await fetch(`${baseUrl}/models`, { headers: makeHeaders(apiKey) });
  if (!response.ok) {
    throw new Error(`Ollama Cloud API returned ${response.status}: ${await response.text()}`);
  }

  const payload = (await response.json()) as {
    data?: Array<{ id: string; object?: string; created?: number; owned_by?: string }>;
  };

  if (!payload.data || !Array.isArray(payload.data)) {
    throw new Error("Unexpected response format from /v1/models");
  }

  return payload.data.map((m) => resolveModelMeta(m.id));
}

// =============================================================================
// Extension Entry Point
// =============================================================================

export default async function (pi: ExtensionAPI) {
  const authStorage = AuthStorage.create();
  const baseUrl = process.env.OLLAMA_CLOUD_BASE_URL || "https://ollama.com/v1";
  const apiKey = await authStorage.getApiKey("ollama-cloud") ?? process.env.OLLAMA_API_KEY;
  const envModels = process.env.OLLAMA_CLOUD_MODELS;
  const timeout = parseInt(process.env.OLLAMA_CLOUD_TIMEOUT || "30000", 10);

  let models: ModelMeta[];
  let discoveryError: string | undefined;

  // ─── Dynamic Discovery ────────────────────────────────────────────────
  try {
    if (envModels) {
      models = envModels.split(",").map((id) => resolveModelMeta(id.trim()));
    } else {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeout);
      try {
        models = await fetchOllamaModels(baseUrl, apiKey);
      } finally {
        clearTimeout(timer);
      }
    }
  } catch (error) {
    discoveryError = error instanceof Error ? error.message : String(error);
    // Fallback: curated list of popular / useful cloud models
    models = [
      resolveModelMeta("llama3.3"),
      resolveModelMeta("llama3.2"),
      resolveModelMeta("llama3.2-vision"),
      resolveModelMeta("llama3.1"),
      resolveModelMeta("llama3.1:70b"),
      resolveModelMeta("qwen3"),
      resolveModelMeta("qwen3:30b"),
      resolveModelMeta("qwen2.5:72b"),
      resolveModelMeta("codestral"),
      resolveModelMeta("deepseek-coder-v2"),
      resolveModelMeta("deepseek-r1"),
      resolveModelMeta("deepseek-r1:70b"),
      resolveModelMeta("gemma3:27b"),
      resolveModelMeta("gemma3:12b"),
      resolveModelMeta("phi4"),
      resolveModelMeta("gpt-oss:120b-cloud"),
      resolveModelMeta("command-r-plus"),
      resolveModelMeta("mistral-large"),
      resolveModelMeta("mixtral:8x22b"),
    ];
  }

  // ─── Register Provider ──────────────────────────────────────────────────
  registerProviderWithKey(pi, baseUrl, models);

  // ─── Startup Notification + API Key Prompt ──────────────────────────────
  pi.on("session_start", async (_event, ctx) => {
    if (discoveryError) {
      ctx.ui.notify(
        `Ollama Cloud: model discovery failed (${discoveryError}). Using fallback list.`,
        "warning",
      );
    } else {
      ctx.ui.notify(
        `Ollama Cloud: ${models.length} model(s) available. Use /model to select.`,
        "info",
      );
    }

    // Prompt for API key if not set (checks auth.json first, then env var)
    const storedKey = await authStorage.getApiKey("ollama-cloud");
    if (!storedKey) {
      const key = await ctx.ui.input(
        "Ollama Cloud API Key",
        "Paste your API key from ollama.com/settings:",
      );
      if (key?.trim()) {
        authStorage.set("ollama-cloud", { type: "api_key", key: key.trim() });
        ctx.ui.notify("API key saved. Run /model to select an ollama-cloud model.", "info");
      } else {
        ctx.ui.notify(
          "Ollama Cloud: API key not set. Set OLLAMA_API_KEY or use /login → API key.",
          "warning",
        );
      }
    }
  });

  // ─── Commands ─────────────────────────────────────────────────────────

  // /ollama-cloud-refresh
  pi.registerCommand("ollama-cloud-refresh", {
    description: "Refresh Ollama Cloud model list from the API",
    handler: async (_args, ctx) => {
      ctx.ui.notify("Refreshing Ollama Cloud models...", "info");
      try {
        const fresh = await fetchOllamaModels(baseUrl, apiKey);
        registerProviderWithKey(pi, baseUrl, fresh);
        ctx.ui.notify(`Ollama Cloud: refreshed ${fresh.length} models.`, "info");
      } catch (error) {
        ctx.ui.notify(
          `Ollama Cloud refresh failed: ${error instanceof Error ? error.message : String(error)}`,
          "error",
        );
      }
    },
  });

  // /ollama-cloud-status
  pi.registerCommand("ollama-cloud-status", {
    description: "Show Ollama Cloud provider status",
    handler: async (_args, ctx) => {
      const storedKey = await authStorage.getApiKey("ollama-cloud");
      const keySource = storedKey
        ? "stored in auth.json"
        : "NOT SET — set OLLAMA_API_KEY or use /login → API key";
      ctx.ui.notify(
        `Ollama Cloud — baseUrl: ${baseUrl}, API key: ${keySource}, models: ${models.length}`,
        storedKey ? "info" : "warning",
      );
    },
  });

  // /ollama-cloud-list
  pi.registerCommand("ollama-cloud-list", {
    description: "List all available Ollama Cloud models",
    handler: async (_args, ctx) => {
      const families = new Map<string, ModelMeta[]>();
      for (const m of models) {
        const f = m.family || "other";
        if (!families.has(f)) families.set(f, []);
        families.get(f)!.push(m);
      }
      const lines: string[] = [`Ollama Cloud Models (${models.length} total)`, "─".repeat(50)];
      for (const [family, familyModels] of families) {
        lines.push(`\n${family.toUpperCase()} (${familyModels.length})`);
        for (const m of familyModels) {
          const cap = m.input.includes("image") ? "🖼️" : "💬";
          const reason = m.reasoning ? "🧠" : "  ";
          lines.push(`  ${reason} ${cap} ${m.name} (${m.id}) — ${m.contextWindow.toLocaleString()} ctx`);
          if (m.description) lines.push(`     └─ ${m.description}`);
        }
      }
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });

  // /ollama-cloud-pull <model>
  pi.registerCommand("ollama-cloud-pull", {
    description: "Pull a model from Ollama Hub (native API)",
    handler: async (args, ctx) => {
      const modelId = args?.trim();
      if (!modelId) {
        ctx.ui.notify("Usage: /ollama-cloud-pull <model-id>", "error");
        return;
      }
      ctx.ui.notify(`Pulling ${modelId}...`, "info");
      // Note: Ollama Cloud doesn't have a pull endpoint; pulling is for local Ollama.
      // This command is a placeholder for future native API integration.
      ctx.ui.notify(
        `Pull is a local Ollama operation. Run: ollama pull ${modelId}`,
        "info",
      );
    },
  });

  // ─── Custom Tools ─────────────────────────────────────────────────────

  // Tool: ollama_list_models
  pi.registerTool({
    name: "ollama_list_models",
    label: "List Ollama Cloud Models",
    description: "List available models on the connected Ollama Cloud instance. Returns model IDs, names, capabilities, and context windows.",
    promptSnippet: "List available Ollama Cloud models with their specs",
    promptGuidelines: [
      "Use ollama_list_models when the user asks what models are available on Ollama Cloud.",
      "Use ollama_list_models before recommending a specific Ollama Cloud model.",
    ],
    parameters: Type.Object({
      filter: Type.Optional(Type.String({ description: "Optional family filter (e.g., 'llama', 'qwen', 'deepseek')" })),
      include_vision: Type.Optional(Type.Boolean({ description: "Only return vision-capable models" })),
      include_reasoning: Type.Optional(Type.Boolean({ description: "Only return reasoning-capable models" })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      let filtered = models;
      if (params.filter) {
        const f = params.filter.toLowerCase();
        filtered = filtered.filter((m) => (m.family || "").toLowerCase().includes(f) || m.id.toLowerCase().includes(f));
      }
      if (params.include_vision) {
        filtered = filtered.filter((m) => m.input.includes("image"));
      }
      if (params.include_reasoning) {
        filtered = filtered.filter((m) => m.reasoning);
      }
      const list = filtered.map((m) => ({
        id: m.id,
        name: m.name,
        family: m.family,
        capabilities: {
          text: m.input.includes("text"),
          vision: m.input.includes("image"),
          reasoning: m.reasoning,
        },
        context_window: m.contextWindow,
        max_tokens: m.maxTokens,
        description: m.description,
      }));
      return {
        content: [{ type: "text", text: JSON.stringify(list, null, 2) }],
        details: { count: list.length, total_available: models.length },
      };
    },
  });

  // Tool: ollama_embeddings
  pi.registerTool({
    name: "ollama_embeddings",
    label: "Ollama Cloud Embeddings",
    description: "Generate text embeddings via Ollama Cloud's /v1/embeddings endpoint. Supports nomic-embed-text, mxbai-embed-large, bge-m3, and other embedding models.",
    promptSnippet: "Generate text embeddings using Ollama Cloud",
    promptGuidelines: [
      "Use ollama_embeddings when the user needs vector embeddings for text.",
      "Use ollama_embeddings for RAG, similarity search, or semantic clustering.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Embedding model ID (e.g., 'nomic-embed-text', 'mxbai-embed-large', 'bge-m3')" }),
      input: Type.Union([
        Type.String({ description: "Single text to embed" }),
        Type.Array(Type.String(), { description: "Multiple texts to embed" }),
      ]),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const toolKey = await authStorage.getApiKey("ollama-cloud") ?? process.env.OLLAMA_API_KEY;
      const body = {
        model: params.model,
        input: Array.isArray(params.input) ? params.input : [params.input],
      };
      const response = await fetch(`${baseUrl}/embeddings`, {
        method: "POST",
        headers: makeHeaders(toolKey),
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const err = await response.text();
        return {
          content: [{ type: "text", text: `Embedding failed: ${response.status} ${err}` }],
          details: { status: response.status },
          isError: true,
        };
      }
      const data = await response.json();
      return {
        content: [{ type: "text", text: JSON.stringify(data, null, 2) }],
        details: { model: params.model, count: Array.isArray(params.input) ? params.input.length : 1 },
      };
    },
  });

  // Tool: ollama_chat
  pi.registerTool({
    name: "ollama_chat",
    label: "Ollama Cloud Chat",
    description: "Send a direct chat completion request to Ollama Cloud via the OpenAI-compatible /v1/chat/completions endpoint. Useful for one-off completions or testing model behavior.",
    promptSnippet: "Send a direct chat request to Ollama Cloud",
    promptGuidelines: [
      "Use ollama_chat for one-off completions or testing a specific Ollama Cloud model.",
      "Use ollama_chat when the user wants to compare outputs across different Ollama models.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Model ID (e.g., 'llama3.3', 'qwen3', 'deepseek-r1')" }),
      messages: Type.Array(
        Type.Object({
          role: Type.String({ description: "Role: system, user, or assistant" }),
          content: Type.String({ description: "Message content" }),
        }),
        { description: "Chat messages" },
      ),
      temperature: Type.Optional(Type.Number({ description: "Sampling temperature (0-2)", default: 0.7 })),
      max_tokens: Type.Optional(Type.Number({ description: "Max tokens to generate" })),
      stream: Type.Optional(Type.Boolean({ description: "Stream response (default false)", default: false })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const body = {
        model: params.model,
        messages: params.messages,
        temperature: params.temperature ?? 0.7,
        max_tokens: params.max_tokens,
        stream: params.stream ?? false,
      };
      const toolKey = await authStorage.getApiKey("ollama-cloud") ?? process.env.OLLAMA_API_KEY;
      const response = await fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: makeHeaders(toolKey),
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const err = await response.text();
        return {
          content: [{ type: "text", text: `Chat failed: ${response.status} ${err}` }],
          details: { status: response.status },
          isError: true,
        };
      }
      const data = await response.json();
      const text = data.choices?.[0]?.message?.content || JSON.stringify(data, null, 2);
      return {
        content: [{ type: "text", text }],
        details: { model: params.model, usage: data.usage },
      };
    },
  });

  // Tool: ollama_model_info
  pi.registerTool({
    name: "ollama_model_info",
    label: "Ollama Model Info",
    description: "Get detailed metadata about a specific Ollama Cloud model including context window, capabilities, and description.",
    promptSnippet: "Get detailed info about an Ollama Cloud model",
    promptGuidelines: [
      "Use ollama_model_info when the user asks about a specific model's capabilities.",
      "Use ollama_model_info to verify context window or vision support before using a model.",
    ],
    parameters: Type.Object({
      model_id: Type.String({ description: "Model ID (e.g., 'llama3.3', 'qwen3:30b')" }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const meta = resolveModelMeta(params.model_id);
      const found = models.find((m) => m.id === params.model_id);
      const info = {
        id: meta.id,
        name: meta.name,
        family: meta.family,
        description: meta.description,
        capabilities: {
          text: meta.input.includes("text"),
          vision: meta.input.includes("image"),
          reasoning: meta.reasoning,
        },
        context_window: meta.contextWindow,
        max_tokens: meta.maxTokens,
        registered: !!found,
        available_in_cloud: !!found,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(info, null, 2) }],
        details: info,
      };
    },
  });
}
