import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import shutil
import threading
import tempfile
import zipfile
from collections import Counter

tf.config.set_visible_devices([], 'GPU')
try:
    tf.config.set_visible_devices([], 'MPS')
except:
    pass

from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import Tokenizer
import os
from datetime import datetime

TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "minigpt_tokenizer (1).json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "minigpt.keras")
LOG_PATH = os.path.join(os.path.dirname(__file__), "training_log.txt")

# Thread lock to prevent concurrent training/generation races (re-entrant)
_model_lock = threading.RLock()

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

# Current target block size for large prompts
TARGET_BLOCK_SIZE = 512
# We must start with 128 to load the original weights safely
block_size = 128
n_embd = 256
n_head = 4
n_layer = 6
dropout = 0.1

class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=n_head,
            key_dim=n_embd // n_head,
            dropout=dropout
        )
        self.proj = layers.Dense(n_embd)
        self.dropout_layer = layers.Dropout(dropout)
        self.ln = layers.LayerNormalization()

    def call(self, x):
        attn_output = self.attention(
            x,
            x,
            use_causal_mask=True
        )
        attn_output = self.dropout_layer(attn_output)
        return self.ln(x + attn_output)

class FeedForward(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = keras.Sequential([
            layers.Dense(4 * n_embd, activation="gelu"),
            layers.Dense(n_embd),
            layers.Dropout(dropout)
        ])

    def call(self, x):
        return self.net(x)

class Block(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = layers.LayerNormalization()
        self.atten = SelfAttention()
        self.ln2 = layers.LayerNormalization()
        self.ffn = FeedForward()

    def call(self, x):
        attn_output = self.atten(self.ln1(x))
        x = x + attn_output
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output
        return x

class MiniGPT(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_vocab_size = vocab_size
        self.current_block_size = block_size

        self.token_embedding = layers.Embedding(
            self.current_vocab_size,
            n_embd,
            name="embedding"
        )
        self.position_embedding = layers.Embedding(
            self.current_block_size,
            n_embd,
            name="position_embedding"
        )
        self.blocks = [
            Block()
            for _ in range(n_layer)
        ]
        self.ln_f = layers.LayerNormalization(name="layer_normalization_f")
        self.lm_head = layers.Dense(self.current_vocab_size, name="lm_head")

    def call(self, idx):
        B = tf.shape(idx)[0]
        T = tf.shape(idx)[1]
        tok_emb = self.token_embedding(idx)
        pos = tf.range(T)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def safe_save_model(model_to_save):
    """Saves model weights to a temp file first, then replaces to avoid 0-byte corruption."""
    try:
        # Save weights to a proper .weights.h5 temp file inside the project dir
        project_dir = os.path.dirname(MODEL_PATH)
        temp_path = os.path.join(project_dir, "_temp_model_save.keras")
        model_to_save.save(temp_path)
        shutil.move(temp_path, MODEL_PATH)
    except Exception as e:
        print("Failed to save model:", e)
        # Clean up temp file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


def _load_weights_from_keras_zip(model, keras_path):
    """
    Extract the .h5 weights file from inside a .keras zip archive and load it.
    This handles the case where load_weights() fails on the zip format.
    """
    if not os.path.exists(keras_path):
        return False

    # First check if it's actually a zip file
    if not zipfile.is_zipfile(keras_path):
        print(f"Warning: {keras_path} is not a valid zip file.")
        return False

    try:
        with zipfile.ZipFile(keras_path, 'r') as zf:
            names = zf.namelist()
            h5_files = [n for n in names if n.endswith('.h5')]
            if not h5_files:
                print("No .h5 file found inside .keras archive.")
                return False

            # Extract the h5 file to a temporary location inside the project
            project_dir = os.path.dirname(keras_path)
            temp_h5 = os.path.join(project_dir, "_temp_weights.h5")
            with zf.open(h5_files[0]) as src, open(temp_h5, 'wb') as dst:
                dst.write(src.read())

        # Load from the extracted h5 file
        model.load_weights(temp_h5)
        print("Successfully loaded pre-trained weights from .keras archive.")

        # Clean up temp file
        if os.path.exists(temp_h5):
            os.remove(temp_h5)
        return True

    except Exception as e:
        print(f"Could not load weights from .keras zip: {e}")
        # Clean up
        temp_h5 = os.path.join(os.path.dirname(keras_path), "_temp_weights.h5")
        if os.path.exists(temp_h5):
            try:
                os.remove(temp_h5)
            except:
                pass
        return False


def initialize_model():
    global vocab_size, tokenizer, block_size
    
    # Try to initialize with the target block size first to avoid mismatch
    # if the model was already expanded.
    current_attempt_block_size = TARGET_BLOCK_SIZE
    print(f"Initializing model with vocab_size={vocab_size} and block_size={current_attempt_block_size}...")
    
    block_size = current_attempt_block_size
    model = MiniGPT()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    dummy_input = tf.zeros((1, 1), dtype=tf.int32)
    model(dummy_input)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_weights(MODEL_PATH)
            print("Successfully loaded pre-trained weights.")
            return model
        except Exception as e:
            print(f"Direct load failed with {current_attempt_block_size}: {e}")
            
            # If 512 failed, maybe it's still 128?
            if current_attempt_block_size != 128:
                print("Retrying with block_size=128...")
                block_size = 128
                model = MiniGPT()
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                )
                model(tf.zeros((1, 1), dtype=tf.int32))
                try:
                    model.load_weights(MODEL_PATH)
                    print("Successfully loaded pre-trained weights with block_size=128.")
                    return model
                except Exception as e2:
                    print(f"Retry with 128 also failed: {e2}")
    
    print("Starting with fresh (untrained) weights.")
    return model

model = initialize_model()


def resize_model_architecture(new_vocab_size, new_block_size):
    global model, vocab_size, block_size

    old_vocab_size = model.current_vocab_size
    old_block_size = model.current_block_size

    if new_vocab_size <= old_vocab_size and new_block_size <= old_block_size:
        return

    print(f"Resizing architecture. Vocab: {old_vocab_size}->{new_vocab_size}, Block: {old_block_size}->{new_block_size}")

    # Vocabulary weights
    old_emb_weights = model.token_embedding.get_weights()[0]
    old_lm_head_weights = model.lm_head.get_weights()[0]
    old_lm_head_biases = model.lm_head.get_weights()[1]

    new_emb_weights = tf.keras.initializers.GlorotUniform()(shape=(new_vocab_size, n_embd)).numpy()
    new_emb_weights[:old_vocab_size, :] = old_emb_weights

    new_lm_head_weights = tf.keras.initializers.GlorotUniform()(shape=(n_embd, new_vocab_size)).numpy()
    new_lm_head_weights[:, :old_vocab_size] = old_lm_head_weights

    new_lm_head_biases = tf.zeros((new_vocab_size,)).numpy()
    new_lm_head_biases[:old_vocab_size] = old_lm_head_biases

    # Position weights
    old_pos_weights = model.position_embedding.get_weights()[0]
    new_pos_weights = tf.keras.initializers.GlorotUniform()(shape=(new_block_size, n_embd)).numpy()
    copy_len = min(old_block_size, new_block_size)
    new_pos_weights[:copy_len, :] = old_pos_weights[:copy_len, :]

    # Build new model
    vocab_size = new_vocab_size
    block_size = new_block_size

    new_model = MiniGPT()
    dummy_input = tf.zeros((1, 1), dtype=tf.int32)
    new_model(dummy_input)

    # Transfer all weights
    new_model.token_embedding.set_weights([new_emb_weights])
    new_model.position_embedding.set_weights([new_pos_weights])
    new_model.lm_head.set_weights([new_lm_head_weights, new_lm_head_biases])
    new_model.ln_f.set_weights(model.ln_f.get_weights())
    for old_block, new_block in zip(model.blocks, new_model.blocks):
        new_block.set_weights(old_block.get_weights())

    new_model.compile(
        optimizer=model.optimizer,
        loss=model.loss
    )

    model = new_model
    safe_save_model(model)
    print("Resized architecture successfully saved.")


# Check and expand block size immediately on startup
if model.current_block_size < TARGET_BLOCK_SIZE:
    resize_model_architecture(vocab_size, TARGET_BLOCK_SIZE)


def ensure_special_tokens():
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "<sos>", "<eos>"]
    added = tokenizer.add_special_tokens(special_tokens)
    if added > 0:
        tokenizer.save(TOKENIZER_PATH)
        new_vocab_size = tokenizer.get_vocab_size()
        if new_vocab_size > vocab_size:
            resize_model_architecture(new_vocab_size, block_size)

ensure_special_tokens()


def update_tokenizer_with_text(text):
    global vocab_size, block_size
    words = text.split()
    new_tokens = []
    for word in words:
        if tokenizer.token_to_id(word) is None:
            new_tokens.append(word)
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        tokenizer.save(TOKENIZER_PATH)
        new_vocab_size = tokenizer.get_vocab_size()
        if new_vocab_size > vocab_size:
            resize_model_architecture(new_vocab_size, block_size)


def generate_text(prompt, max_new_tokens=50, temperature=0.8):
    with _model_lock:
        update_tokenizer_with_text(prompt)

        prompt_tokens = tokenizer.encode(prompt).ids
        SOS_ID = tokenizer.token_to_id("<sos>")
        EOS_ID = tokenizer.token_to_id("<eos>")
        PAD_ID = tokenizer.token_to_id("[PAD]") or 0

        # FIX: operator precedence — parenthesise the ternary correctly
        if SOS_ID is not None:
            tokens = [SOS_ID] + prompt_tokens.copy()
        else:
            tokens = prompt_tokens.copy()

        new_tokens = []

        # Strong repetition penalty
        repetition_penalty = 3.0
        hard_block_window = 8
        hard_block_max = 2

        for _ in range(max_new_tokens):
            context = tokens[-block_size:]
            x = tf.constant([context], dtype=tf.int32)
            logits = model(x, training=False)
            logits = logits[:, -1, :]
            logits_np = logits.numpy()

            # Hard-ban PAD and special tokens from generation
            logits_np[0, PAD_ID] = -1e9
            if SOS_ID is not None:
                logits_np[0, SOS_ID] = -1e9

            # Apply repetition penalty on recent tokens
            recent = new_tokens[-hard_block_window:]
            recent_counts = Counter(recent)
            for token_id, count in recent_counts.items():
                if count >= hard_block_max:
                    logits_np[0, token_id] = -1e9  # Hard ban
                else:
                    if logits_np[0, token_id] > 0:
                        logits_np[0, token_id] /= repetition_penalty
                    else:
                        logits_np[0, token_id] *= repetition_penalty

            logits_tensor = tf.constant(logits_np) / temperature
            probs = tf.nn.softmax(logits_tensor)
            next_token = int(tf.random.categorical(
                tf.math.log(probs),
                num_samples=1
            )[0, 0].numpy())

            if EOS_ID is not None and next_token == EOS_ID and len(new_tokens) > 2:
                break

            tokens.append(next_token)
            new_tokens.append(next_token)

        return tokenizer.decode(new_tokens)


def train_on_text(text):
    # Enforce eager in this thread
    if not tf.executing_eagerly():
        try:
            tf.compat.v1.enable_eager_execution()
        except:
            pass

    with _model_lock:
        # 1. Update tokenizer and resize model if needed
        grew = update_tokenizer_with_text(text)
        if grew:
            safe_save_model(model)

        SOS_ID = tokenizer.token_to_id("<sos>")
        EOS_ID = tokenizer.token_to_id("<eos>")
        PAD_ID = tokenizer.token_to_id("[PAD]") or 0

        raw_tokens = tokenizer.encode(text).ids
        tokens = []
        if SOS_ID is not None:
            tokens.append(SOS_ID)
        tokens.extend(raw_tokens)
        if EOS_ID is not None:
            tokens.append(EOS_ID)

        if len(tokens) < 2:
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SKIPPED | Text: \"{text[:60]}\" | Tokens: {len(tokens)} (too short)\n"
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)
            return

        # 2. Create sequences with overlap
        x_data = []
        y_data = []
        stride = max(1, block_size // 2)
        for i in range(0, len(tokens) - 1, stride):
            end = min(i + block_size + 1, len(tokens))
            chunk = tokens[i:end]
            if len(chunk) >= 2:
                x_data.append(chunk[:-1])
                y_data.append(chunk[1:])

        if not x_data:
            return

        # 3. Manual padding (plain numpy)
        max_len = block_size # Always pad to context window for consistency
        x_padded = np.full((len(x_data), max_len), PAD_ID, dtype=np.int32)
        y_padded = np.full((len(y_data), max_len), PAD_ID, dtype=np.int32)
        for idx, (xseq, yseq) in enumerate(zip(x_data, y_data)):
            l = min(len(xseq), max_len)
            x_padded[idx, :l] = xseq[:l]
            y_padded[idx, :l] = yseq[:l]

        # 4. Training loop
        epochs = 30
        batch_size = 8
        total_loss = 0.0
        steps = 0

        print(f"Training started on text: \"{text[:50]}\"...")
        for epoch in range(epochs):
            indices = np.arange(len(x_padded))
            np.random.shuffle(indices)
            x_shuffled = x_padded[indices]
            y_shuffled = y_padded[indices]

            epoch_loss = 0.0
            for i in range(0, len(x_shuffled), batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                loss = model.train_on_batch(x_batch, y_batch)
                total_loss += float(loss)
                epoch_loss += float(loss)
                steps += 1
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/(len(x_shuffled)/batch_size + 1):.4f}")

        avg_loss = total_loss / steps if steps > 0 else 0.0
        print(f"Training complete. Avg Loss: {avg_loss:.4f}")

        log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TRAINED | Text: \"{text[:60]}\" | Tokens: {len(tokens)} | Chunks: {len(x_data)} | Epochs: {epochs} | Avg Loss: {avg_loss:.4f}\n"
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_entry)

        # 5. Save weights
        safe_save_model(model)
        return avg_loss
