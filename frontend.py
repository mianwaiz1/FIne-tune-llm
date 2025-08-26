# streamlit_chat_app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =======================
# Load fine-tuned model
# =======================
@st.cache_resource
def load_model():
    model_path = "./fine_tuned_lung_cancer"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# =======================
# Chat function
# =======================
def chat(question):
    inputs = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Lung Cancer Chatbot", page_icon="💬", layout="centered")

st.title("💬 Lung Cancer Chatbot")
st.write("Ask me anything about **lung cancer**. (Fine-tuned model)")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Chat input
if prompt := st.chat_input("Type your question..."):
    # Save user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate bot response
    response = chat(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
