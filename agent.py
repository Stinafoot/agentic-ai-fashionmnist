"""
agent.py — Agentic AI Workflow for FashionMNIST classification
Uses LangChain + Ollama (llama3.2 / mistral / phi3) with 4 tools:
  1. load_image_tool
  2. predict_class_tool
  3. confidence_check_tool
  4. explanation_tool

Run:  python agent.py
Requires: Ollama running locally with a model pulled (see README)
"""

import io
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import TypedDict, Annotated, Optional
from operator import add

# LangChain / LangGraph imports 
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# SECTION 1 — MODEL DEFINITION (must match train.py exactly)

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

CONFIDENCE_THRESHOLDS = {
    "high":   0.85,
    "medium": 0.60,
    "low":    0.0,
}

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Load model once at module level 
_model = FashionCNN()
try:
    _model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    _model.eval()
    print("[agent] model.pt loaded successfully.")
except FileNotFoundError:
    print("[agent] WARNING: model.pt not found. Run train.py first.")

_preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])


# SECTION 2 — TOOL DEFINITIONS

@tool
def load_image_tool(image_path: str) -> str:
    """
    Load an image from disk and validate it is a readable greyscale image.
    Returns a JSON string with status and image metadata, or an error message.
    Input: image_path (str) — absolute or relative path to a PNG/JPG file.
    """
    try:
        img = Image.open(image_path).convert("L")
        width, height = img.size
        return json.dumps({
            "status": "success",
            "path": image_path,
            "width": width,
            "height": height,
            "mode": "L",
            "message": f"Image loaded: {width}x{height} greyscale.",
        })
    except FileNotFoundError:
        return json.dumps({
            "status": "error",
            "path": image_path,
            "message": f"File not found: '{image_path}'. Check the path and try again.",
        })
    except Exception as exc:
        return json.dumps({
            "status": "error",
            "path": image_path,
            "message": f"Could not open image: {exc}",
        })


@tool
def predict_class_tool(image_path: str) -> str:
    """
    Run the FashionMNIST CNN on the image at image_path.
    Returns a JSON string with predicted class, class index, and full probability distribution.
    Input: image_path (str) — path to a PNG/JPG file (should be pre-validated by load_image_tool).
    """
    try:
        img    = Image.open(image_path).convert("L")
        tensor = _preprocess(img).unsqueeze(0)          # (1,1,28,28)
        with torch.no_grad():
            logits = _model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze()
        idx   = int(probs.argmax())
        label = CLASSES[idx]
        conf  = float(probs[idx])
        dist  = {CLASSES[i]: round(float(probs[i]), 4) for i in range(10)}
        return json.dumps({
            "status":        "success",
            "predicted_class": label,
            "class_index":   idx,
            "confidence":    round(conf, 4),
            "all_probabilities": dist,
        })
    except FileNotFoundError:
        return json.dumps({"status": "error", "message": f"File not found: '{image_path}'"})
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def confidence_check_tool(confidence: float) -> str:
    """
    Interpret a raw confidence score (0.0–1.0) and return a human-readable tier.
    Returns a JSON string with tier ('high', 'medium', or 'low') and interpretation.
    Input: confidence (float) — the top-class probability from predict_class_tool.
    """
    if confidence >= CONFIDENCE_THRESHOLDS["high"]:
        tier = "high"
        interpretation = (
            f"The model is highly confident ({confidence:.1%}). "
            "This prediction is very reliable."
        )
    elif confidence >= CONFIDENCE_THRESHOLDS["medium"]:
        tier = "medium"
        interpretation = (
            f"The model has moderate confidence ({confidence:.1%}). "
            "The prediction is plausible but consider verifying with a second opinion."
        )
    else:
        tier = "low"
        interpretation = (
            f"The model has low confidence ({confidence:.1%}). "
            "The image may be ambiguous, corrupted, or unlike anything in the training set."
        )
    return json.dumps({
        "status":         "success",
        "confidence":     round(confidence, 4),
        "tier":           tier,
        "interpretation": interpretation,
    })


@tool
def explanation_tool(predicted_class: str, confidence: float, confidence_tier: str) -> str:
    """
    Produce a structured natural-language explanation of the prediction result.
    Input:
      predicted_class  (str)   — class label from predict_class_tool
      confidence       (float) — raw confidence score
      confidence_tier  (str)   — 'high', 'medium', or 'low' from confidence_check_tool
    Returns a JSON string with a human-readable explanation and recommendation.
    """
    category_descriptions = {
        "T-shirt/top":  "a short-sleeved upper-body garment",
        "Trouser":      "long-leg lower-body clothing",
        "Pullover":     "a long-sleeved knitted upper garment",
        "Dress":        "a one-piece garment covering the torso and legs",
        "Coat":         "a heavy outer garment for cold weather",
        "Sandal":       "an open footwear with straps",
        "Shirt":        "a collared upper-body garment with buttons",
        "Sneaker":      "a casual athletic shoe",
        "Bag":          "a carried accessory for storing items",
        "Ankle boot":   "a short boot covering the ankle",
    }
    desc = category_descriptions.get(predicted_class, "a clothing/accessory item")

    recommendations = {
        "high":   "No further verification needed.",
        "medium": "Consider reviewing the image quality or providing a clearer sample.",
        "low":    "Manual review is strongly recommended — the model is uncertain.",
    }

    explanation = (
        f"The image has been classified as '{predicted_class}' — {desc}. "
        f"The model assigned this label with {confidence:.1%} confidence, "
        f"which is considered {confidence_tier}. "
        f"{recommendations.get(confidence_tier, '')}"
    )

    return json.dumps({
        "status":         "success",
        "predicted_class": predicted_class,
        "confidence":     round(confidence, 4),
        "confidence_tier": confidence_tier,
        "explanation":    explanation,
        "recommendation": recommendations.get(confidence_tier, ""),
    })


# SECTION 3 — AGENT STATE


class AgentState(TypedDict):
    messages:          Annotated[list, add]   # full message history
    image_path:        Optional[str]
    predicted_class:   Optional[str]
    confidence:        Optional[float]
    confidence_tier:   Optional[str]
    final_explanation: Optional[str]
    error:             Optional[str]



# SECTION 4 — LANGGRAPH WORKFLOW


TOOLS = [load_image_tool, predict_class_tool, confidence_check_tool, explanation_tool]

# Connect to local Ollama — change model name to match what you have pulled
llm = ChatOllama(model="llama3.2", temperature=0).bind_tools(TOOLS)


def agent_node(state: AgentState) -> AgentState:
    """The LLM reasoning node — decides which tool to call next."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Route: if the last message has tool calls → execute tools; else → done."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(TOOLS)


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()



# SECTION 5 — RUNNER HELPERS

SYSTEM_PROMPT = """You are an AI assistant specialized in clothing image classification.
You have access to four tools that you MUST use in this exact sequence for every request:
  1. load_image_tool    — validate the image exists and is readable
  2. predict_class_tool — run the CNN classifier on the image
  3. confidence_check_tool — interpret the confidence score
  4. explanation_tool   — produce the final structured explanation

Always use all four tools before giving your final answer.
If load_image_tool returns an error, report the error immediately and do not call the other tools.
"""

def run_agent(image_path: str) -> dict:
    """Run the agent workflow for a single image path and return the final state."""
    graph = build_graph()
    user_message = (
        f"Classify this image and explain the prediction confidence. "
        f"Image path: {image_path}"
    )
    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=SYSTEM_PROMPT + "\n\nUser request: " + user_message)
        ],
        "image_path":        image_path,
        "predicted_class":   None,
        "confidence":        None,
        "confidence_tier":   None,
        "final_explanation": None,
        "error":             None,
    }
    final_state = graph.invoke(initial_state)
    return final_state


def print_result(state: dict, case_label: str):
    print(f"\n{'='*60}")
    print(f"  {case_label}")
    print(f"{'='*60}")
    messages = state.get("messages", [])
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        if isinstance(msg, HumanMessage):
            # Only print first 120 chars of the prompt
            print(f"[{role}] {str(msg.content)[:120]}…")
        elif isinstance(msg, AIMessage):
            if msg.content:
                print(f"[{role}] {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  → Tool call: {tc['name']}({json.dumps(tc['args'])})")
        elif isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                print(f"  ← Tool result ({msg.name}): {json.dumps(data, indent=4)}")
            except Exception:
                print(f"  ← Tool result: {msg.content}")
    print()


# SECTION 6 — DEMO: SUCCESS CASE + FAILURE CASE

if __name__ == "__main__":
    import os, urllib.request

    # Success case: download one FashionMNIST test image 
    SAMPLE_IMAGE = "sample_sneaker.png"
    if not os.path.exists(SAMPLE_IMAGE):
        print("Downloading sample image …")
        try:
            # Pull a sample from torchvision and save it as a PNG
            from torchvision import datasets
            import torchvision.transforms.functional as TF
            ds = datasets.FashionMNIST(root="./data", train=False, download=True,
                                       transform=transforms.ToTensor())
            # Find a sneaker (class 7)
            for img_tensor, label in ds:
                if label == 7:
                    pil_img = TF.to_pil_image(img_tensor)
                    pil_img.save(SAMPLE_IMAGE)
                    print(f"Saved sample sneaker → {SAMPLE_IMAGE}")
                    break
        except Exception as e:
            print(f"Could not download sample: {e}")
            SAMPLE_IMAGE = "sample_sneaker.png"  # will trigger file-not-found 

    # Case 1: Valid image
    print("\n>>> CASE 1: Valid image")
    state1 = run_agent(SAMPLE_IMAGE)
    print_result(state1, "CASE 1 — SUCCESS (valid image)")

    # Case 2: Invalid path 
    print("\n>>> CASE 2: Invalid image path (failure case)")
    state2 = run_agent("/nonexistent/path/to/image.png")
    print_result(state2, "CASE 2 — FAILURE (invalid image path)")
