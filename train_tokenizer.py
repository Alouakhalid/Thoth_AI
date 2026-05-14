from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import os

dataset_path = os.path.join(os.path.dirname(__file__), "user_dataset.txt")
tokenizer_path = os.path.join(os.path.dirname(__file__), "minigpt_tokenizer (1).json")

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()


trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "<sos>", "<eos>"])

tokenizer.train([dataset_path], trainer)

tokenizer.save(tokenizer_path)
print(f"Tokenizer saved to {tokenizer_path} with vocab size: {tokenizer.get_vocab_size()}")
