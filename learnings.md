# Learnings

## Definitions
- Tokens: Tokens are the units of text a model reads and writes. They are not always whole words: a tokenizer may split a word into smaller pieces, while keeping punctuation or common words as single tokens.
- MarianMT checkpoint: A saved, pretrained machine translation model, including its learned weights and the settings needed to use it. For example, Helsinki-NLP/opus-mt-en-de is a checkpoint for translating English to German.
- Encoder-decoder Transformer: A model architecture with two parts. The encoder reads the input sentence and builds a representation of it; the decoder uses that representation to generate the translated sentence, token by token.
