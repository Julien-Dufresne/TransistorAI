
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Rather than manually changing the model ID every time, you can create a dictionary:
MODELS = {
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
}

# =======> Change LANGUAGE_PAIR to "en-de" whenever you want English → German. Remember to choose the corresponding test sentences and direction-specific output filenames.
LANGUAGE_PAIR = "de-en"

MODEL_ID = MODELS[LANGUAGE_PAIR]

# Selecting the Hardware: Use Apple Silicon acceleration if available
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Load the tokenizer and model. AutoTokenizer loads the model's text processing configuration and AutoModelForSeq25SeqLM loads the pretrained Transformer weights
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# .to(DEVICE) places the model on your selected CPU or accelerator.
model = model.to(DEVICE)

# .eval() switches the model to inference mode, disabling training-specific behavior such as dropout.
model.eval()

print(f"Loaded model: {MODEL_ID}")

# ==========================
# Creating the first translation function
import time

DECODING_CONFIG = {
    "num_beams": 4, # Beam search considers multiple candidate translations
    "do_sample": False, # Uses deterministic decoding rather than random sampling
    "max_new_tokens": 128, # Limits output length
}

# Measuring GPU execution correctly
def synchronize_device():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def translate(text: str) -> dict:
    """Translate one English sentence into German."""

# The tokenizer will split the sentence into tokens, convert them to IDs, and return a tensor suitable for the model. The truncation and max_length parameters ensure that the input does not exceed the model's maximum sequence length.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE) # moves the resulting tensors to the same device as the model.

    synchronize_device() # calculate the time taken for tokenization and moving tensors to the device before starting the timer.
    start = time.perf_counter()

# torch.inference_mode() tells PyTorch that you're performing inference rather than training. Gradient calculations aren't required, which reduces overhead.
    with torch.inference_mode():
        generated_ids = model.generate( #runs the model's autoregressive generation process.
            **inputs,
            **DECODING_CONFIG,
        )

    synchronize_device()
    latency_ms = (time.perf_counter() - start) * 1000

# converts the generated token IDs into strings.
    translation = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

# Rather than simply returning translated text, your function returns a structured dictionary.
    return {
        "source": text,
        "translation": translation,
        "model_id": MODEL_ID,
        "device": DEVICE,
        "decoding": DECODING_CONFIG.copy(),
        "latency_ms": round(latency_ms, 2),
    }

# ==========================
# We will now test sentences and measure the latency of the translation process. 
# The results will be saved into two files: day2_baseline.csv with the translations and latency measurements, and day2_config.json with the experimental settings.
import pandas as pd
import json
from pathlib import Path

TEST_SENTENCES = [
    "Das maschinelle Lernmodell sagt das nächste Wort voraus.",
    "Die Anwendung ist nach dem letzten Softwareupdate abgestürzt.",
    "Wir haben das Transformer-Modell mit domänenspezifischen Daten feinabgestimmt.",
    "Das Modell erzielte eine höhere Genauigkeit, benötigte jedoch mehr Speicher.",
    "Der Benutzer muss sich anmelden, bevor er auf verschlüsselte Daten zugreifen kann.",
    "Die Bank lehnte die Anfrage ab, weil das Konto inaktiv war.",
]

if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Warm up the model before measuring test sentences
    translate("This is a warm-up sentence.")

    results = []

    for sentence in TEST_SENTENCES:
        result = translate(sentence)
        results.append(result)

        print(f"\nEN: {result['source']}")
        print(f"DE: {result['translation']}")
        print(f"Latency: {result['latency_ms']} ms")

    df = pd.DataFrame(results)

    
    # ==========================================
    # STEP 6: Prepare manual evaluation file
    # ==========================================

    evaluation_path = Path("evaluation/manual_evaluation_de_en.csv")
    evaluation_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Give every sentence a stable ID
    df.insert(
        0,
        "id",
        [f"T{i:03d}" for i in range(1, len(df) + 1)]
    )

    # Add empty columns for your manual scores
    for column in [
        "meaning",
        "terminology",
        "fluency",
        "completeness",
        "total_score",
        "error_category",
        "notes",
    ]:
        df[column] = ""

    # Create the file without deleting past evaluations
    if evaluation_path.exists():
        print("Evaluation file already exists. Skipping.")
    else:
        df.to_csv(evaluation_path, index=False)
        print(f"Created {evaluation_path}")


    df.to_csv(
        output_dir / "day2_baseline_de_en.csv",
        index=False,
    )

    with open(output_dir / "day2_config.json", "w") as f:
        json.dump(
            {
                "model_id": MODEL_ID,
                "device": DEVICE,
                "decoding": DECODING_CONFIG,
                "torch_version": torch.__version__,
            },
            f,
            indent=2,
        )

    print("\nAverage latency:",
          round(df["latency_ms"].mean(), 2), "ms")