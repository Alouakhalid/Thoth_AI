import requests
import time
import os
import random

def load_user_dataset():
    path = "user_dataset.txt"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        # Split by empty lines or standard paragraph breaks
        paras = [p.strip() for p in content.split("\n") if len(p.strip()) > 30]
        # Remove duplicates to avoid over-focusing
        paras = list(set(paras))
        return paras

# Core knowledge paragraphs
core_knowledge = [
    "Thoth is the ancient Egyptian god of wisdom, writing, and magic. He is often depicted with the head of an ibis.",
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals and humans.",
    "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.8+ based on standard Python type hints.",
    "The Transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data.",
    "The Nile is a major north-flowing river in northeastern Africa, and is commonly regarded as the longest river in the world.",
    "Mathematics is an area of knowledge that includes the study of numbers, formulas, and related structures, shapes and spaces."
]

# Load user history
user_data = load_user_dataset()
all_data = core_knowledge + user_data

# Shuffle to prevent the model from getting stuck in patterns (helps prevent forgetting)
random.shuffle(all_data)

print(f"--- THOTH KNOWLEDGE REINFORCEMENT ---")
print(f"Total paragraphs to review: {len(all_data)}")
print(f"Total estimated training steps: {len(all_data) * 30} epochs")
print("=" * 60)

# We run 2 passes to reinforce the knowledge
for pass_num in range(1, 3):
    print(f"\n>>> PASS {pass_num} STARTING...")
    random.shuffle(all_data)
    
    for i, para in enumerate(all_data):
        # Only show preview for every 10th or so to keep console clean
        if i % 5 == 0 or len(all_data) < 10:
            print(f"[{i+1}/{len(all_data)}] Training on: \"{para[:60]}...\"")
        
        try:
            response = requests.post(
                "http://127.0.0.1:8000/train",
                json={"text": para},
                timeout=120
            )
            if response.status_code != 200:
                print(f"  ✗ Error: {response.text}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # Small sleep to let the server breathe
        time.sleep(0.1)

print("\n" + "=" * 60)
print("Reinforcement complete! Thoth's memory has been strengthened.")
