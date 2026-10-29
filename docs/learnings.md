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

## Evaluations metrics. 
### BLEU: Bilingual Evaluation Understudy:  
BLEU evaluates how much a machine-generated translation overlaps with one or more human reference translations. It examines matching word sequences called n-grams.

**Strengths of BLEU**. 

BLEU is fast, reproducible and widely used in machine translation research.
It is particularly useful for comparing model performance on a fixed evaluation dataset.
Limitations of BLEU

**How BLEU works**
BLEU compares n-grams of different lengths.

**Strengths of BLEU**
BLEU is fast, reproducible and widely used in machine translation research.
It is particularly useful for comparing model performance on a fixed evaluation dataset.
**Limitations of BLEU**
Its main weakness is that it does not directly understand meaning.
It can penalize valid translations with different wording and sometimes reward incorrect translations containing many matching words.
Also, standard BLEU is designed primarily for evaluation over a corpus of sentences, rather than judging one individual sentence.

### chrF: Character n-gram F-score:  
chrF works similarly to BLEU, but it compares sequences of characters instead of words.
This is particularly useful for morphologically rich languages such as German.
**How chrF works**
chrF uses character-level precision and recall.
Precision: How much of the generated translation's character n-gram content matches the reference?
Recall: How much of the reference's character n-gram content is covered by the generated translation?  

**Strengths of chrF**. 
- Relatively simple to calculate.
- Language-independent character matching.
- Useful for languages with complex morphology.
- Often complements BLEU well.
- Less sensitive to some tokenization choices.
**Limitations of chrF**
chrF still measures surface similarity rather than meaning.

### COMET: Crosslingual Optimized Metric for Evaluation of Translation:  
COMET is fundamentally different from BLEU and chrF.
Instead of comparing only text fragments, COMET uses a trained neural model to estimate translation quality.  
**How COMET works**
Reference-based COMET typically receives three inputs:
1. Source sentence: The original text.
2. Machine translation: The generated translation.
3. Reference: A human translation.

**Strengths of COMET**
- Better equipped to assess semantic similarity.
- Can evaluate meaning preservation.
- Often correlates more closely with human judgments than overlap-only metrics.
- Useful for identifying subtle translation quality differences.
- Reference-free variants can support interactive applications.
**Limitations of COMET**
- Computationally more expensive.
- Requires downloading and executing a neural evaluation model.
- Scores are model-dependent and not necessarily percentages.
- May exhibit biases.
- Can overlook severe individual errors.
- Requires careful evaluation of the evaluator itself.

## Libraries we will use:
**PyTorch**: Runs the neural network. 
**Transformers**: Load the pretrained model and generate translations
**SentencePiece**: Tokenization required by Marian
**Sacremoses**: Text Preprocessing dependencies
**Pandas** Store and analyze experimental results

# Baseline model
We will use Helsinki-NLP/opus-mt-en-de which is a good baseline model for translation. It uses MarianMT, a Transformer encoder–decoder architecture trained for English-to-German translation

# Conda environment: 
It is an isolated workspace on my computer where we can install a specific version of Python and the libraries needed for a project. It is important to keep project independant from each other. 
- Project folder >>> Stores my code, datasets, notebooks and results
- Conda environment >>> Provides Python and installed libraries
- VS Code >>> Editor where we write and run codes
- GitHub >>> Stores and versions our project files

-- 22.10.26 --

# OPUS KDE4 dataset: 
It is a widely used multilingual parallel corpus consisting of localized text and documentation from the KDE4 open-source desktop environment package. It serves as a foundational benchmark dataset for training and fine-tuning Machine Translation (MT) models, specifically within technical or software localization domains.
It features 92 languages and counts over 224.7 million tokens overall.

# OPUS GNOME dataset:
It is a massive, open-source multilingual parallel corpus created by compiling localized interface text, user manuals, and system documentation from the GNOME desktop environment package.
It is highly technical data consisting of software strings (buttons, drop-down menus, system messages) and comprehensive desktop user manuals. This makes it rich in technical terminology and concise, actionable phrasing.
It features 100s of languages and includes 1.7 billion sentences. 

# unicodedata module
It is a built-in library in Python that provides direct access to the official Unicode Character Database (UCD). It allows to query character names, determine character properties (for example if a character is a digit or a symbol), and normalize text to prevent issues with characters that look identical but have different code points. 

# assert statement with Python:
It is a statement to perform data integrity checks and it tests a condition: 
- if the condition evaluates to True, the code silently moves to the next line
- if the condition evaluates to False, the code crashes immediately with an AssertionError, halting the pipeline

# out-of-domain evaluation
Also called OOD evaluation, is a method used in Machine Learning to test how well a model performs on data that is different from the data it was trained on. The goal being to measure the model's ability to generalize to unfamiliar domains, rather than simply perform well on familiar data. 

## Day 5 - Understand the evaluation metrics

# BLEU: Bilingual Evaluation Understudy
Measures how much the model's translation overlaps with a human reference translation, using word sequences (n-grams) and a brevity penalty.
Useful for comparing translation systems on the same dataset.

# chrF — Character n-gram F-score
Measures character-sequence overlap between the generated translation and the reference.
Particularly useful for German, where compound words and inflected forms can make exact word matching difficult.

## Day 6: 

# Tokenization during translation: 
A translation model does not directly process words when translating. Instead, it processes numerical representations called token IDs. Example: "Der Motor funktionert nicht" => "The motor is not working"
'__Der' (ID 1), '_Motor' (ID 2), '_functioniert' (ID 3), '_nicht' (ID 4), '.' (ID 5)   

For supervised ML, the tokenizer prepares two different elements: 
- input_ids: tokens IDs representing the German source sentence
- labels: token IDs representing the expected English translation

Tokenization target becomes labels during training. 

# inputs_ids: What the model reads
It is what the model reads. They are the numerical IDs representing the source sentence. Each ID corresponds to a token in the tokenizer's vocabulary. A token might represent a whole word, part of a word, punctuation or a special symbol. Neural networks operate on numerical representations rather than raw text. 

The model converts token IDs into learned vector representation called embeddings. 

# attention_mask Which positions contain real tokens

Sentence: "Der Motor funktioniert."
**input_ids**:      125, 842, 394, 5, 0, 58100, 58100, 58100

**attention_mask**: 1,   1,   1,   1, 1, 0,     0,     0

1 = real token, 0 = padding

As sentences have different lengths, when preparing a batch, the tokenizer or data collator may add padding tokens so the tensors have consistent dimensions. 

The model uses the attention mask to prevent padding positions from influencing its attention calculations.

# labels What the model should learn to predict
labels contain the token IDs of the correct English translation.

During training, the model compares its predictions with these labels and calculates a loss.

A subtle but important point: training loss is not calculated by directly comparing two finished text strings. It is generally calculated using token-level cross-entropy between predicted probabilities and the reference token IDs.

When labels are padded, DataCollatorForSeq2Seq typically replaces padded label positions with -100, which tells PyTorch to ignore them when calculating loss.

# text_target Telling the tokenizer to process target text

text_target is not a tensor or model input key. It is an argument of the Hugging Face tokenizer.

It tells the tokenizer that the provided sentences are target translations.
This matters because some sequence-to-sequence tokenizers apply different processing rules to source and target languages, such as language-specific special tokens.

For your German–English MarianMT model, you use it to tokenize the English reference sentences appropriately.

# Understanding the hyperparameters, including initial values in (), and purpose:
- Seed (42):                Makes random operations more reproducible
- Optimizer (adamw_torch):  Algorithm that adjusts model weights
- Learning rate (2e-5):     How large the weight updates are
- Batch size (4) :          Number of translation pairs processed per device in one forward/backward step
- Gradient accumulation (2): Combines gradients from 2 batches before updating weights
- Epochs (3):               Number of complete passes through training data
- Max source length (128):  Maximum input tokens
- Max target length (128):  Maximum reference translation tokens during training
- Checkpoint (Every epoch): Saves model state for recovery or comparison

