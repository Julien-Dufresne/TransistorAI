# import the libraries and define paths
# using Path ensures that the paths work when you run the script from your project root

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'technical' / 'de-en'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'technical' / 'de-en'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# select the files to load as the EN or the DE documents

EN_FILE = RAW_DIR / 'KDE4.de-en.en'
DE_FILE = RAW_DIR / 'KDE4.de-en.de'

# load and align English-German sentences. This is critical as parallel corpora depend on alignment. 
# If English line 25 does not correspond to German line 25, the model may learn incorrect translations.

# **IMPORTANT** Do not remove empty lines from either language file independently before alignment.

with open(DE_FILE, encoding="utf-8") as f:
    german = f.read().splitlines()

with open(EN_FILE, encoding="utf-8") as f:
    english = f.read().splitlines()

print("English lines:", len(english))
print("German lines:", len(german))

if len(english) != len(german):
    raise ValueError("The parallel files have different line counts!")

# Create the dataframe with English sentences as source and German sentences as target
df = pd.DataFrame({
    "source": english,
    "target": german
})

print(df.head())
print("Original sentence pairs:", len(df))

# Normalise the text: Normalize Unicode and removes excessive whitespace without changing the meaning of the sentences.

import unicodedata
from ftfy import fix_text


## To normalize whitespace, we can use the split() method to break the text into words and then join them back together with a single space. This will remove any extra spaces, tabs, or newlines.
def normalize_text(text):

    # Repair common encoding problems
    text = fix_text(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFC", text)

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    return text

df["source"] = df["source"].map(normalize_text)
df["target"] = df["target"].map(normalize_text)

# Remove empty and duplicate pairs

before = len(df)

# creates a fresh copy of the dataframe filtering out any rows where either the source or target is an empty string. This ensures that we only keep valid sentence pairs for training.
df = df[
    (df["source"] != "") &
    (df["target"] != "")
].copy()

print("Removed empty pairs:", before - len(df))

before = len(df)

# Removing duplicate pairs
df = df.drop_duplicates(
    subset=["source", "target"]
).reset_index(drop=True)

print("Removed duplicate pairs:", before - len(df))

# Filtering out suspicious sentence pairs - as some translation pairs may contain very long text or significantly different lengths
# We will use a 2-200 word filter initially
df["source_words"] = df["source"].str.split().str.len()
df["target_words"] = df["target"].str.split().str.len()

before = len(df)

df = df[
    df["source_words"].between(2, 100) &
    df["target_words"].between(2, 100)
].copy()

print("Removed by length filter:", before - len(df))
print("Remaining pairs:", len(df))


# Manually inspect the dataset by inspecting 30 random examples

sample = df.sample(
    n=min(30, len(df)),
    random_state=42
)

print(sample[["source", "target"]].to_string(index=False))

sample[["source", "target"]].to_csv(
    OUTPUT_DIR / "manual_inspection.csv",
    index=False
)

# Create reproductible dataset splits for training, validation, and testing. We will use a 80-10-10 split.


from sklearn.model_selection import train_test_split

# Group by source text so the same English sentence
# cannot appear in different splits.
unique_sources = df["source"].drop_duplicates()

train_sources, temp_sources = train_test_split(
    unique_sources,
    test_size=0.20,
    random_state=42
)

validation_sources, test_sources = train_test_split(
    temp_sources,
    test_size=0.50,
    random_state=42
)

train = df[df["source"].isin(train_sources)]
validation = df[df["source"].isin(validation_sources)]
technical_test = df[df["source"].isin(test_sources)]

for name, dataset in [
    ("train", train),
    ("validation", validation),
    ("technical_test", technical_test)
]:
    dataset[["source", "target"]].to_csv(
        OUTPUT_DIR / f"{name}.csv",
        index=False
    )
    print(name, len(dataset))

# validate the output using the assert statement. It ensures the dataset split (train, validation and technical test) performed correctly


assert len(train) + len(validation) + len(technical_test) == len(df)

assert set(train["source"]).isdisjoint(validation["source"])
assert set(train["source"]).isdisjoint(technical_test["source"])
assert set(validation["source"]).isdisjoint(technical_test["source"])

for dataset in [train, validation, technical_test]:
    assert not dataset[["source", "target"]].isna().any().any()
    assert not dataset.duplicated(["source", "target"]).any()

print("Dataset validation passed!")
