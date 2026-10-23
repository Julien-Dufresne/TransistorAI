# ============================================================
# TransistorAI - General-Domain Test Dataset Preparation
# Dataset: WMT19 newstest2019 English-German
# Purpose: Out-of-domain machine translation evaluation
# ============================================================

from pathlib import Path
import pandas as pd
import unicodedata
from ftfy import fix_text

# STEP 1 - Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "general"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "general" / "de-en"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EN_FILE = RAW_DIR / "de-en" / "newstest2019.en"
DE_FILE = RAW_DIR / "de-en" / "newstest2019.de"

# STEP 2 - Normalize text

def normalize_text(text):

    # Repair common encoding problems
    text = fix_text(text)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFC", text)

    # Remove unnecessary whitespace
    text = " ".join(text.split())

    return text

# STEP 3 - Load aligned English and German sentences
with open(EN_FILE, encoding="utf-8") as f:
    english = f.read().splitlines()

with open(DE_FILE, encoding="utf-8") as f:
    german = f.read().splitlines()

print(f"English sentences: {len(english)}")
print(f"German sentences: {len(german)}")

if len(english) != len(german):
    raise ValueError("English and German files are not aligned!")



# STEP 4 - Create DataFrame
df = pd.DataFrame({
    "source": english,
    "target": german
})

print(f"Original sentence pairs: {len(df)}")

# STEP 5 - Clean text
df["source"] = df["source"].map(normalize_text)
df["target"] = df["target"].map(normalize_text)

# Remove empty pairs
df = df[
    (df["source"] != "") &
    (df["target"] != "")
].copy()

# Remove exact duplicates
df = df.drop_duplicates(
    subset=["source", "target"]
).reset_index(drop=True)

print(f"Clean sentence pairs: {len(df)}")

# STEP 6 - Save the complete benchmark
output_file = OUTPUT_DIR / "general_test.csv"

df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)

print(f"Saved dataset to: {output_file}")

# STEP 7 - Display sample translations
print("\nSample translations:")
print(df[["source", "target"]].head(10).to_string(index=False))

# STEP 8 - Validate
assert not df.empty
assert not df[["source", "target"]].isna().any().any()
assert not df.duplicated(["source", "target"]).any()

print("\nGeneral-domain test dataset validated successfully!")
