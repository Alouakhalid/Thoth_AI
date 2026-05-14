from tokenizers import Tokenizer, decoders
tokenizer = Tokenizer.from_file("minigpt_tokenizer (1).json")
tokenizer.decoder = decoders.BPEDecoder()
encoded = tokenizer.encode("hello world")
print(f"Tokens: {encoded.tokens}")
print(f"Decoded BPEDecoder: {tokenizer.decode(encoded.ids)}")
