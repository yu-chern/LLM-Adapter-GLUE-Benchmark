# LLM-Adapter-GLUE-Benchmark

## 📌 Overview

This repository benchmarks **eight adapter-based fine-tuning strategies** (using a TinyBERT backbone) across **four GLUE tasks**: SST-2, QNLI, MNLI, and QQP.

### ✅ What This Repo Does:

* Trains the following adapter variants:
  * **Soft Prompt**, **Prefix**, **LoRA**, **Soft Prompt + LoRA**, **Prefix + LoRA**, **(IA)^3**, **Single-Layer Fine-Tuning**, **Full Fine-Tuning**

* Applies the **DP-SGD** optimization algorithm with two privacy budgets:
  * `epsilon = 8` (differentially private training)
  * `epsilon = ∞` (standard, non-private training)

* Evaluates **classification accuracy** of each adapter on the corresponding GLUE validation sets.

### 🧰 Tech Stack:

* **Environment**: Google Colab
* **Frameworks**: `PyTorch`, `Transformers`, `Opacus`, `PEFT`, `huggingface_hub`

---

## 🚀 Getting Started

1. Copy the `llm_adapters_comparison/` folder to your Colab environment.

2. Configure hyperparameters via JSON:
   * `hyper_parameter_config_w_privacy.json` (with DP)
   * `hyper_parameter_config_wo_privacy.json` (without DP)

3. Open and run `main.ipynb` step-by-step:
   * Set `project_root_dir` to the path of the copied folder.
   * Navigate (`cd`) to `project_root_dir`.
   * Generate a Hugging Face access token and store it in Colab secrets.

4. Open the .xlsx files to check the results.
   * The final results are saved in the `output/` folder:
      - `accuracy_results_eps_*.xlsx` saves classification accuracy
      - `parameter_scale_eps_*.xlsx` saves the number of trainable parameters from the adapters

---


