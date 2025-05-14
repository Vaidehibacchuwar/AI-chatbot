import streamlit as st 
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import base64

# Adding background image and styling
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .bold-text {{
        font-weight: 900;  /* Extra bold */
        color: #000000;    /* Deep black color */
        text-align: right;
    }}
    .chatbot-text {{
        color: #000000;    /* Black color for chatbot response */
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function to set the background
set_background(r"C:\codes\December\bert\Chatbots-in-Machine-Learning-scaled.webp")

# Load BERT tokenizer and model
@st.cache_resource
def load_bert_model():
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

tokenizer, model = load_bert_model()

# Predefined questions and responses
qa_pairs = {
    "What is your name?": "I am a chatbot powered by BERT!",
    "How are you?": "I'm just a bunch of code, but I'm doing great!",
    "What is BERT?": "BERT stands for Bidirectional Encoder Representations from Transformers. It’s a powerful NLP model.",
    "Tell me a joke.": "Why don't programmers like nature? It has too many bugs.",
    "what is data science": "**Data Science** is the study of analyzing data to find useful information.",
    "what is your use":"A **BERT-based chatbot** uses the BERT model to understand and respond to user queries.",
    "What is ai":"**Artificial Intelligence (AI)** is the ability of machines to simulate human intelligence.",
    "what is microsoft azure":"**Microsoft Azure** is a cloud computing platform and service provided by Microsoft.",
}

# Function to get BERT embeddings
@st.cache_data
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Precompute embeddings for predefined questions
predefined_embeddings = {question: get_bert_embedding(question) for question in qa_pairs}

# Function to get the chatbot's response
def chatbot_response(user_input):
    user_embedding = get_bert_embedding(user_input)
    
    # Compute cosine similarities
    similarities = {
        question: cosine_similarity(user_embedding, predefined_embeddings[question])[0][0]
        for question in qa_pairs
    }
    
    # Find the most similar question
    best_match = max(similarities, key=similarities.get)
    
    # Return the response if similarity is high enough
    if similarities[best_match] > 0.6:  # Adjusted threshold
        return qa_pairs[best_match]
    else:
        return "I'm not sure how to respond to that."

# Streamlit Frontend
st.markdown("<h1 class='bold-text'>BERT Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p class='bold-text'>This is a BERT-powered chatbot application with a user-friendly interface.</p>", unsafe_allow_html=True)
st.markdown("<h3 class='bold-text'>Ask me anything!</h3>", unsafe_allow_html=True)

# User input with button
user_input = st.text_input("**You:**", placeholder="Type your message here...")
if st.button("Ask") and user_input:
    response = chatbot_response(user_input)
    st.markdown(f"<p class='chatbot-text'><b>Chatbot:</b> {response}</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")