# Learnings

## Definitions
**Tokens**:  
Tokens are the units of text a model reads and writes. They are not always whole words: a tokenizer may split a word into smaller pieces, while keeping punctuation or common words as single tokens.  

**MarianMT checkpoint**:  
A saved, pretrained machine translation model, including its learned weights and the settings needed to use it. For example, Helsinki-NLP/opus-mt-en-de is a checkpoint for translating English to German. 

**Encoder-decoder Transformer**:  
A model architecture with two parts. The encoder reads the input sentence and builds a representation of it; the decoder uses that representation to generate the translated sentence, token by token.

**Transformer**:  
A neural network architecture that processes text using attention. It can model relationships between words or tokens across a sentence, and it powers models such as MarianMT.  

**Attention**:  
A mechanism that lets the model weigh which other tokens are most relevant when interpreting or generating a particular token. In translation, the decoder can use attention to connect a German word it is generating with relevant parts of the English input.
