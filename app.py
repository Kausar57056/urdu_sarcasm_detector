
import streamlit as st
import torch
from transformers import AutoTokenizer
from sentimixturenet import SentimixtureNet

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimixtureNet()
    model.load_state_dict(torch.load("sentimixture_model.pt", map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return model.to(device), tokenizer, device

def run_app():
    st.title("🤖 Urdu Sarcasm Detection")
    st.markdown("Enter an Urdu tweet and I will tell you if it's sarcastic or not.")
    model, tokenizer, device = load_model()

    tweet = st.text_area("✍️ Enter Urdu Tweet:", height=100)
    if st.button("🔍 Predict"):
        if not tweet.strip():
            st.warning("Please enter a tweet to continue.")
            return
        encoding = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output, dim=1).item()
        st.success("😏 Sarcastic!" if prediction == 1 else "🙂 Not Sarcastic.")

run_app()
