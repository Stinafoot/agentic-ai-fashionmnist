# Agentic AI System for FashionMNIST

## Overview
This project implements an agentic AI workflow using LangGraph and a local LLM via Ollama. The system classifies FashionMNIST images using a CNN and enhances predictions with tool-based reasoning and explanation generation.

---

## System Architecture

The agent follows a structured 4-step tool pipeline:

1. Load Image Tool  
   - Validates and reads image

2. Prediction Tool  
   - CNN model classifies image

3. Confidence Tool  
   - Converts probability into confidence level

4. Explanation Tool  
   - Generates natural language reasoning

---

## Tech Stack
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Workflows-purple)
![LangChain](https://img.shields.io/badge/LangChain-LLM_Framework-green)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)

---

## Workflow

User Input → Load Image → CNN Prediction → Confidence Check → Explanation → Final Output

---

## How to Run

### Install dependencies

pip install -r requirements.txt


### Start Ollama

ollama serve
ollama pull llama3.2


### Run agent

python agent.py


---

## Example Output

### Valid Image

Predicted: Sneaker
Confidence: 1.0
Tier: High
Explanation: The model is highly confident...


### Invalid Image

Error: File not found


---

## Key Features
- Multi-tool agent workflow
- Local LLM reasoning (Ollama)
- CNN-based image classification
- Structured explanations
- Error handling

---

## Author
AI/ML Agentic Systems Project
