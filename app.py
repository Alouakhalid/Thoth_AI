from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from model_utils import generate_text, train_on_text
import os
import re
import tensorflow as tf

# Force eager execution for thread safety
try:
    tf.compat.v1.enable_eager_execution()
except:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "index.html not found"}

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 500
    temperature: float = 0.8

class TrainRequest(BaseModel):
    text: str

def filter_english(text: str) -> str:
    return re.sub(r'[^\x20-\x7E\n\t]+', '', text).strip()

def process_background_training(text: str):
    dataset_path = os.path.join(os.path.dirname(__file__), "user_dataset.txt")
    with open(dataset_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    train_on_text(text)

@app.post("/generate")
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    try:
        response_text = generate_text(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature
        )
        
        filtered_response = filter_english(response_text)
        
        background_tasks.add_task(process_background_training, req.prompt)
        
        return {"response": filtered_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train(req: TrainRequest, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(process_background_training, req.text)
        return {"status": "success", "message": "Training started in background"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
def get_logs():
    log_path = os.path.join(os.path.dirname(__file__), "training_log.txt")
    if not os.path.exists(log_path):
        return {"logs": []}
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return {"logs": [line.strip() for line in lines if line.strip()]}

@app.get("/stats")
def get_stats():
    from model_utils import model, tokenizer, block_size, vocab_size
    log_path = os.path.join(os.path.dirname(__file__), "training_log.txt")
    train_count = 0
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            train_count = sum(1 for line in f if "TRAINED" in line)
    total_params = sum(
        int(tf.reduce_prod(v.shape)) for v in model.trainable_variables
    )
    return {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "parameters": total_params,
        "parameters_human": f"{total_params / 1_000_000:.1f}M",
        "train_sessions": train_count,
        "layers": 6,
        "embedding_dim": 256,
        "attention_heads": 4,
        "version": "Thoth V1"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
