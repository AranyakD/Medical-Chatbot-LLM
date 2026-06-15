import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "medquad.csv")

def load_and_clean():
    df = pd.read_csv(DATA_PATH)

    df = df[['question', 'answer']].dropna()

    def clean_answer(text):
        if isinstance(text, str):
            return ". ".join(text.split(".")[:3])
        return text

    df["answer"] = df["answer"].apply(clean_answer)

    df["input_text"] = "Answer medically: " + df["question"]
    df["target_text"] = df["answer"]

    df.to_csv(os.path.join(BASE_DIR, "data", "medquad_cleaned.csv"), index=False)

    print("Data saved successfully!")

if __name__ == "__main__":
    load_and_clean()