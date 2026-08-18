import os
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import requests
from urllib.parse import quote

# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY in .env (or in Render's Environment tab)")
    st.stop()

# Groq is OpenAI SDK-compatible: same client, different base_url
client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

# --- Model / provider config ---
TEXT_MODEL = "openai/gpt-oss-120b"   # free on Groq, no credit card needed
IMAGE_BASE_URL = "https://image.pollinations.ai/prompt/"  # free, no API key needed


def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_image(prompt: str) -> bytes:
    url = IMAGE_BASE_URL + quote(prompt)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


# Streamlit page config
st.set_page_config(page_title="Medical Information Agent", page_icon="🩺")
st.title("🩺 AI Medical Information Agent")
st.markdown("Get symptoms, treatments, medicines, and nutrition advice using trusted sources like WHO, Mayo Clinic, and WebMD.")

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar - user info
with st.sidebar:
    st.header("User Information")
    name = st.text_input("Name", key="name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1, key="age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")

# Tabs
tab1, tab2 = st.tabs(["💬 Chat with Agent", "🦠 Disease Info"])

# --- TAB 1: Chat ---
with tab1:
    st.subheader("Chat with the Medical Agent")
    user_input = st.chat_input("Describe your symptoms or ask a health-related question")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        text_prompt = f"""
        You are a helpful, trustworthy medical assistant AI. Use only verified sources like WHO, Mayo Clinic, and WebMD.
        Patient Name: {name}
        Age: {age}
        Gender: {gender}

        The user asked: {user_input}

        Provide clear, trustworthy, and actionable information. Include:
        - Symptoms
        - Treatments
        - Medicines
        - Nutrition suggestions (if applicable)
        """

        with st.spinner("Generating response..."):
            try:
                response_text = generate_text(text_prompt)
                st.session_state.chat_history.append(("assistant", response_text))
            except Exception as e:
                response_text = f"Error generating medical info: {e}"
                st.session_state.chat_history.append(("assistant", response_text))

    # Display chat history (newest first)
    for role, msg in reversed(st.session_state.chat_history):
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    # Nutrition Image
    if user_input:
        nutrition_prompt = f"Photorealistic nutrition plate for a person with symptoms or condition described as: '{user_input}'. Age: {age}, Gender: {gender}. Based on WHO or Mayo Clinic guidance."

        with st.spinner("Generating nutrition image..."):
            try:
                img_bytes = generate_image(nutrition_prompt)
                st.markdown("<h4 style='text-align:center;'>Nutrition Suggestion</h4>", unsafe_allow_html=True)
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Nutrition image error: {e}")

    # Medicine Image
    if user_input:
        medicine_prompt = f"High-resolution image of common medicines or treatment kits based on symptoms: '{user_input}', using WHO or Mayo Clinic guidance. White background."

        with st.spinner("Generating medicine image..."):
            try:
                med_img_bytes = generate_image(medicine_prompt)
                st.markdown("<h4 style='text-align:center;'>Medicine Reference</h4>", unsafe_allow_html=True)
                st.image(med_img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Medicine image error: {e}")

# --- TAB 2: Disease Info ---
with tab2:
    st.subheader("Get Information About a Disease")
    disease_name = st.text_input("Enter Disease Name", key="disease_name")

    if disease_name:
        disease_prompt = f"""
        You are a trusted medical assistant. Give a comprehensive explanation about the disease: {disease_name}.
        Use only WHO, Mayo Clinic, or WebMD as your references.

        Include:
        - Description of the disease
        - Common symptoms
        - Recommended treatments
        - Suggested medicines (generic if possible)
        - Nutrition advice if relevant
        """

        with st.spinner("Generating disease information..."):
            try:
                st.markdown(generate_text(disease_prompt))
            except Exception as e:
                st.error(f"Error generating disease info: {e}")

        # Nutrition Image
        nutrition_prompt = f"Photorealistic nutrition plate for a person with {disease_name}, based on WHO/Mayo Clinic dietary guidance. Age: {age}, Gender: {gender}."
        with st.spinner("Generating nutrition image..."):
            try:
                img_bytes = generate_image(nutrition_prompt)
                st.markdown("<h4 style='text-align:center;'>Nutrition Suggestion</h4>", unsafe_allow_html=True)
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Nutrition image error: {e}")

        # Medicine Image
        medicine_prompt = f"High-resolution image of common medicines or treatment kits for {disease_name}, recommended by WHO or Mayo Clinic. White background."
        with st.spinner("Generating medicine image..."):
            try:
                med_img_bytes = generate_image(medicine_prompt)
                st.markdown("<h4 style='text-align:center;'>Medicine Reference</h4>", unsafe_allow_html=True)
                st.image(med_img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Medicine image error: {e}")

# Footer
st.markdown("<footer style='text-align:center; margin-top:3rem;'>Powered by Groq (Llama 3.3) & Pollinations Image API</footer>", unsafe_allow_html=True)
