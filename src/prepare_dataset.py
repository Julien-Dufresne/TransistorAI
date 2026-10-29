# ============================================================
# 1. import the libraries and define paths
# using Path ensures that the paths work when you run the script from your project root
# ============================================================

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

# Create the dataframe with German sentences as source and English sentences as target
df = pd.DataFrame({
    "source": german,
    "target": english
})

print(df.head())
print("Original sentence pairs:", len(df))

# ============================================================
# 2. Normalise the text: Normalize Unicode and removes excessive whitespace without changing the meaning of the sentences.
# ============================================================

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

# ============================================================
# 3. Remove empty and duplicate pairs
# ============================================================

before = len(df)

# creates a fresh copy of the dataframe filtering out any rows where either the source or target is an empty string. This ensures that we only keep valid sentence pairs for training.
df = df[
    (df["source"] != "") &
    (df["target"] != "")
].copy()

print("Removed empty pairs:", before - len(df))

before = len(df)

# ============================================================
# 4. Removing duplicate pairs
# ============================================================

df = df.drop_duplicates(
    subset=["source", "target"]
).reset_index(drop=True)

print("Removed duplicate pairs:", before - len(df))

# ============================================================
# 5. Filtering out suspicious sentence pairs - as some translation pairs may contain very long text or significantly different lengths
# We will use a 2-100 word filter initially
# ============================================================

df["source_words"] = df["source"].str.split().str.len()
df["target_words"] = df["target"].str.split().str.len()

before = len(df)

df = df[
    df["source_words"].between(2, 100) &
    df["target_words"].between(2, 100)
].copy()

print("Removed by length filter:", before - len(df))
print("Remaining pairs:", len(df))



# ============================================================
# 6. A. Detection of metadata contamination, placeholder mismatches, and identical source-target pairs. This function can be used to flag potential quality issues in the dataset.
# ============================================================





import re

def detect_quality_issues(row):
    issues = []

    source = str(row["source"])
    target = str(row["target"])

    # 1. Detect KDE metadata contamination
    
    metadata_patterns = [
    # KDE annotations
    r"@[A-Za-z]+:",
    r"@action\b",
    r"@label\b",

    # Existing extraction artifacts
    r"GenericName$",
    r"NAME OF TRANSLATORS",
    r"ROLES_OF_TRANSLATORS",
    r"amount in units \(integer\)",
    r"the unit 1 per second",
    r"\.kgm$",
    r"Widget name\. This string",

    # Newly discovered artifacts
    r"No value has been set$",
    r"default price mode$",
    r"City in Newfoundland Canada$",
    r"show help$",

    # Appended KDE Name field
    r"(?<=[a-zäöüß])Name$",
]

    # Inspect both German source and English target.
# Metadata artifacts may occur in either language.
    if any(
        re.search(pattern, text, re.IGNORECASE)
        for text in (source, target)
        for pattern in metadata_patterns
    ):
        issues.append("metadata_contamination")

    # 2. Detect placeholder mismatches
    placeholder_pattern = r"%\d+"

    source_placeholders = sorted(
        re.findall(placeholder_pattern, source)
    )

    target_placeholders = sorted(
        re.findall(placeholder_pattern, target)
    )

    if source_placeholders != target_placeholders:
        issues.append("placeholder_mismatch")

    # 3. Detect identical source and target
    if source == target:
        issues.append("identical_source_target")

    # 4. Detect suspicious sentence length ratios
    source_words = len(source.split())
    target_words = len(target.split())

    if source_words > 0:
        length_ratio = target_words / source_words

        if length_ratio < 0.2 or length_ratio > 2.5:
            issues.append("extreme_length_ratio")

    return ";".join(issues)




#
# ============================================================
# 6.B. QUALITY DETECTION AND FILTERING
# ============================================================

# STEP 1 — Detect potential quality issues in each translation pair.
# The function returns semicolon-separated flags, or an empty
# string if no issues were detected.
df["quality_flags"] = df.apply(
    detect_quality_issues,
    axis=1
)

# STEP 2 — Display the distribution of detected quality issues.
print("\nQuality flags before filtering:")
print(df["quality_flags"].value_counts())

original_count = len(df)

# STEP 3 — Save all flagged translation pairs for manual review.
# This includes potentially valid identical translations.
flagged_df = df[
    df["quality_flags"] != ""
].copy()

flagged_df.to_csv(
    OUTPUT_DIR / "flagged_quality_review.csv",
    index=False,
    encoding="utf-8"
)

print(f"\nSaved {len(flagged_df):,} flagged pairs for review.")

# STEP 4 — Define which quality issues should trigger exclusion.
# Identical source-target pairs are deliberately NOT included,
# because technical terms, commands, and product names can
# legitimately be identical in German and English.
removal_flags = {
    "metadata_contamination",
    "placeholder_mismatch",
    "extreme_length_ratio",
}

# STEP 5 — Check whether a row contains an exclusion flag.
def should_remove(flags):
    # Convert the semicolon-separated string into a set.
    detected_flags = set(flags.split(";")) if flags else set()

    # Return True if any detected flag requires exclusion.
    return bool(detected_flags & removal_flags)


# STEP 6 — Identify translation pairs to exclude.
removal_mask = df["quality_flags"].map(should_remove)

# Save excluded pairs separately for reproducibility.
removed_df = df.loc[removal_mask].copy()

removed_df.to_csv(
    OUTPUT_DIR / "removed_quality_flags.csv",
    index=False,
    encoding="utf-8"
)

# STEP 7 — Keep translation pairs that do not meet
# the configured exclusion criteria.
df = df.loc[~removal_mask].copy()

# STEP 8 — Calculate the filtering results.
removed_count = original_count - len(df)

# STEP 9 — Remove the temporary quality column.
df = df.drop(columns=["quality_flags"])

# STEP 10 — Reset the DataFrame index.
df = df.reset_index(drop=True)

# STEP 11 — Verify that no exclusion flags remain.
assert not df.apply(
    lambda row: should_remove(detect_quality_issues(row)),
    axis=1
).any()

print("\nQuality filtering results:")
print(f"Original pairs: {original_count:,}")
print(f"Removed pairs: {removed_count:,}")
print(f"Remaining pairs: {len(df):,}")
print("Quality filtering completed successfully!")


# 6. C. Manually inspect the dataset by inspecting 30 random examples

sample = df.sample(
    n=min(30, len(df)),
    random_state=42
)

print(sample[["source", "target"]].to_string(index=False))

sample[["source", "target"]].to_csv(
    OUTPUT_DIR / "manual_inspection.csv",
    index=False
)



# 7. Create reproductible dataset splits for training, validation, and testing. We will use a 80-10-10 split.


from sklearn.model_selection import train_test_split

# Group by German source text.
# This prevents the same German source sentence
# from appearing in multiple dataset splits.
#
# All corresponding English translations of that
# German sentence remain in the same split.
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

# python src/prepare_dataset.py



