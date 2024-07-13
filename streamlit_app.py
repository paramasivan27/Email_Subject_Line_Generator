import streamlit as st
from transformers import pipeline
import os

hf_api_key = os.getenv("HUGGING_FACE_API_KEY")
# Load the summarization pipeline with your pre-trained model
pipe = pipeline("summarization", model="paramasivan27/bart_for_email_summarization_enron")

# Function to summarize email
def summarize_email(email_body, pipeline):
    # Tokenize the input text
    print("Tokenizer")
    input_tokens = pipeline.tokenizer(email_body, return_tensors='pt', truncation=True, padding='max_length', max_length=1024)
    input_length = input_tokens['input_ids'].shape[1]
    
    # Adjust max_length to be a certain percentage of the input length
    adjusted_max_length = max(10, int(input_length * 0.6))  # Ensure a minimum length
    
    # Generate summary with dynamic max_length
    gen_kwargs = {
        "length_penalty": 2.0,
        "num_beams": 4,
        "max_length": adjusted_max_length,
        "min_length": 3
    }
    
    summary = pipeline(email_body, **gen_kwargs)[0]['summary_text']
    return summary


# Streamlit UI
st.title("Email Subject Line Generator")

# Text area to input the email body
email_body = st.text_area("Enter the email body:", height=300)

# Button to generate subject line
if st.button("Generate Subject Line"):
    if email_body:
        print("before calling summarize")
        summary = summarize_email(email_body, pipe)
        print("after calling summarize")
        st.subheader("Generated Subject Line:")
        st.write(summary)
    else:
        st.warning("Please enter the email body to generate a subject line.")
