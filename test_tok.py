from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("minigpt_tokenizer (1).json")
encoded = tokenizer.encode("hello world")
print(f"Encoded tokens: {encoded.tokens}")
print(f"Decoded: {tokenizer.decode(encoded.ids)}")
