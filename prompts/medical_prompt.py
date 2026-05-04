def build_prompt(user_input):
    return f"""
You are a medical assistant.

Answer ONLY the final medical explanation.
Do not repeat instructions.
Do not repeat the question.
Do not output rules.

Question: {user_input}

Answer:
"""