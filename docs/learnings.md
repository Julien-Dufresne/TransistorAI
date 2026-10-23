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