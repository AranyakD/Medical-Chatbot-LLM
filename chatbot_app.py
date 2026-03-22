from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./medical-chatbot-model-kaggle"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device("cpu")
model.to(device)

DISCLAIMER = "This is for general guidance only. Consult a doctor for professional advice."


def ask_bot(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            early_stopping=True
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer + "\n\n" + DISCLAIMER


if __name__ == "__main__":
    print("Medical Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = ask_bot(user_input)
        print("Bot:", response)
