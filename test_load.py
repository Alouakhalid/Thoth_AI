import tensorflow as tf
from model_utils import MiniGPT, Block, SelfAttention, FeedForward

try:
    model = tf.keras.models.load_model(
        "minigpt.keras",
        custom_objects={
            "MiniGPT": MiniGPT,
            "Block": Block,
            "SelfAttention": SelfAttention,
            "FeedForward": FeedForward
        },
        compile=False
    )
    print("SUCCESS: Model loaded with load_model")
except Exception as e:
    print("FAILED:", e)
