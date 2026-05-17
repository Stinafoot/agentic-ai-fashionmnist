# Assignment 7 part 2 - Agentic AI System for FashionMNIST

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
- PyTorch
- LangGraph
- LangChain
- Ollama (local LLM)
- Python

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
