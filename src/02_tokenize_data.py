import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "medquad_cleaned.csv")

#load data
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

#convert to hugging face
dataset = Dataset.from_pandas(df)

#Load tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#function
def tokenize_function(example):
    model_inputs = tokenizer(
        example["input_text"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["target_text"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#Apply tokenization
tokenized_dataset = dataset.map(tokenize_function)

# Saving data
SAVE_PATH = os.path.join(BASE_DIR, "data", "tokenized_dataset")

tokenized_dataset.save_to_disk(SAVE_PATH)

print("Tokenization complete. Saved at:", SAVE_PATH)