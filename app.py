import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from prompts.medical_prompt import build_prompt
import torch

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# Page Config
st.set_page_config(
    page_title="Medical Chatbot",    
    layout="wide",
    initial_sidebar_state="collapsed"
)

#UI
st.markdown("""
<div style='max-width: 750px; margin: auto; padding-top: 40px;'>
""", unsafe_allow_html=True)

st.title(" Medical Chatbot")
st.markdown("Ask any medical-related question and get an AI-generated response.")

# Load Model (cached)
@st.cache_resource
def load_model():
    model_path = "../code/medical-chatbot-model-kaggle"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Chat Input
user_input = st.text_input("Ask a medical question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking"):
            prompt = build_prompt(user_input)
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt"                
            ).to(device)           

            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=180,
                    num_beams=5,                    
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True           
                )

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            answer = answer.replace("This is an AI-generated response", "") 
        
        st.markdown(f"""
        <div style="
            background-color:#1f3b2c;
            padding:15px;
            border-radius:12px;
            margin-top:20px;\
            color:white;
            font-size:16px;
            line-height:1.6;
        ">
        {answer}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("This is an AI-generated response. Not a substitute for medical advice.")