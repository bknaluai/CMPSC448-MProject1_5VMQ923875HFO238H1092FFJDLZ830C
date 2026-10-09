# LLM Fingerprinting: Project Repository

**CMPSC 448: Machine Learning — Midterm Project 1**  
**Author:** Blake Naluai  

## Repository Overview

This repository contains the dataset generation scripts, PyTorch model implementations, and technical report for the LLM Fingerprinting project. For the complete theoretical background, methodology, and detailed analysis of findings, please refer to `llm_report_REV2.pdf`.

## File Descriptions

* **`llm_report_REV2.pdf`**  
  The full research report documenting project objectives, model architecture design, experimental methodology, and detailed evaluation across all four research questions.

* **`midterm_project1_REV1.py`**  
  Main Python execution script. Implements PyTorch `TextCNN` and `TextRNN` (BiLSTM) models, custom Dataset/DataLoader pipelines, vocabulary generation, training loops, evaluation metrics, and the experimental logic for RQ1 through RQ4.

* **`load_existing_dataset_REV1.py`**  
  Data extraction and preprocessing script. Fetches raw conversation data from the LMSYS Chatbot Arena dataset (`agie-ai/lmsys-chatbot_arena_conversations`), filters and maps target models into standard family labels (Anthropic, Meta, OpenAI), tags prompt task types, and outputs the balanced dataset.

* **`llm_dataset.csv`**  
  The curated CSV dataset containing $1,500$ balanced samples ($500$ per LLM family) with fields for prompts, model outputs, model family labels, and assigned task categories.

## Setup and Running

1. **Install Dependencies:**
   ```bash
   pip install torch pandas numpy scikit-learn datasets
   ```

2. **Generate or Refresh Dataset:**
   ```bash
   python load_existing_dataset_REV1.py
   ```

3. **Run Model Training & Experiments:**
   ```bash
   python midterm_project1_REV1.py
   ```
