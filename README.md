# Fine-Tuning Llama 3.2 3B with R1-Style Reasoning

This project fine-tunes **Llama 3.2 3B** to generate **structured reasoning-style outputs** inspired by **DeepSeek R1-style thinking**, using **QLoRA**, **PEFT**, **TRL**, and **bitsandbytes** on **Google Colab**.

The goal of this project was to teach a compact instruction-tuned LLM to produce more reflective, step-by-step reasoning traces before arriving at a final answer, while keeping training efficient enough to **run** on limited GPU resources.

---

## Project Overview

Most instruction-tuned LLMs can answer questions directly, but they do not always produce explicit reasoning in a structured way. In this project, I fine-tuned **Llama 3.2 3B** on the **ServiceNow-AI/R1-Distill-SFT** dataset so that the model learns to generate more chain-of-thought-style reasoning traces inspired by **DeepSeek R1-style thinking**.

Instead of full fine-tuning, I used **QLoRA** to reduce memory usage and train only lightweight **LoRA adapters**, making the workflow practical on **consumer or Colab GPUs**.

---

## Features

- Fine-tuned **Llama 3.2 3B Instruct** for reasoning-style text generation
- Used **QLoRA** for memory-efficient 4-bit training
- Added **LoRA adapters** to attention and MLP projection layers
- Trained using **TRL SFTTrainer**
- Formatted dataset into prompt-based reasoning examples
- Saved and reloaded LoRA adapters for inference
- Designed to run on **Google Colab**

---

## Tech Stack

- **Python**
- **PyTorch**
- **Transformers**
- **Datasets**
- **TRL**
- **PEFT**
- **bitsandbytes**

---

## Dataset

**Dataset available on Hugging Face:**  
[`ServiceNow-AI/R1-Distill-SFT`](https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT)

It includes 3 fields:
- `problem`
- `reannotated_assistant_content`
- `solution`

These fields were combined into a single reasoning-style training sequence so the model could learn to generate structured thinking before giving the final answer.

---

## Base Model

**Model used:**  
`unsloth/Llama-3.2-3B-Instruct`

The base model was loaded in **4-bit NF4 quantized form** and then wrapped with **LoRA adapters** for efficient fine-tuning.

---

## Work Pipeline

The full workflow looks like this:

1. Load tokenizer and configure padding
2. Load the R1-style dataset from Hugging Face
3. Format each example into a reasoning prompt
4. Load the base Llama model in **4-bit quantized mode**
5. Prepare the model for **k-bit training**
6. Attach **LoRA adapters**
7. Fine-tune using **SFTTrainer**
8. Save LoRA adapters and tokenizer
9. Reload the adapters for inference

---

## Training Configuration

Key training settings:

- **Trainer:** `SFTTrainer`
- **Max sequence length:** `2048`
- **Batch size:** `2`
- **Gradient accumulation steps:** `4`
- **Learning rate:** `2e-4`
- **Optimizer:** `adamw_8bit`
- **Weight decay:** `0.01`
- **Scheduler:** `linear`
- **Max steps:** `50` *(can be increased for better results)*

---


## Inference Workflow

To run inference after training:

1. Load the original base model in the same 4-bit quantized format
2. Load the tokenizer
3. Attach the trained adapters`
4. Provide a reasoning-style prompt
5. Generate the response

This allows the trained reasoning behavior to be reused without saving the full merged model.

---

## How to Run

### 1. Install dependencies

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

### 2. Load the dataset and tokenizer

Use the Hugging Face dataset and tokenizer for the chosen model.

### 3. Run training

Train using the formatted dataset, QLoRA configuration, and `SFTTrainer`.

### 4. Save adapters

Save the LoRA adapters and tokenizer after training.

### 5. Run inference

Reload the base model, attach the adapters, and generate reasoning-style outputs.

---


## Disclaimer

This project encourages reasoning-style outputs, but generated reasoning traces should still be validated, especially for important or sensitive tasks.

---

## 📄 License
MIT

