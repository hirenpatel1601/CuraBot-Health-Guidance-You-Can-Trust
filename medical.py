import os
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types
from io import BytesIO
from PIL import Image
import base64

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY in .env")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# --- Model names (update this section as Google rotates models) ---
TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"
# gemini-2.5-flash is scheduled to shut down Oct 16, 2026 -> migrate to gemini-3.5-flash before then
# gemini-2.5-flash-image is scheduled to shut down Oct 2, 2026 -> migrate to gemini-3.1-flash-image-preview before then

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
                text_response = client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=[text_prompt]
                )
                response_text = text_response.text
                st.session_state.chat_history.append(("assistant", response_text))
            except Exception as e:
                response_text = f"Error generating medical info: {e}"
                st.session_state.chat_history.append(("assistant", response_text))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

    # Nutrition Image
    if user_input:
        nutrition_prompt = f"Photorealistic nutrition plate for a person with symptoms or condition described as: '{user_input}'. Age: {age}, Gender: {gender}. Based on WHO or Mayo Clinic guidance."

        with st.spinner("Generating nutrition image..."):
            try:
                nutrition_response = client.models.generate_content(
                    model=IMAGE_MODEL,
                    contents=nutrition_prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                )
                img_bytes = next(
                    (part.inline_data.data for part in nutrition_response.candidates[0].content.parts if part.inline_data and part.inline_data.data),
                    None
                )
                if img_bytes:
                    img = Image.open(BytesIO(img_bytes))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    st.markdown("<h4 style='text-align:center;'>Nutrition Suggestion</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{img_str}' style='max-width:100%; border-radius:15px; box-shadow: 0 10px 20px rgba(0,0,0,0.2);'/></div>", unsafe_allow_html=True)
                else:
                    st.warning("No nutrition image received.")
            except Exception as e:
                st.error(f"Nutrition image error: {e}")

    # Medicine Image
    if user_input:
        medicine_prompt = f"High-resolution image of common medicines or treatment kits based on symptoms: '{user_input}', using WHO or Mayo Clinic guidance. White background."

        with st.spinner("Generating medicine image..."):
            try:
                med_response = client.models.generate_content(
                    model=IMAGE_MODEL,
                    contents=medicine_prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                )
                med_img_bytes = next(
                    (part.inline_data.data for part in med_response.candidates[0].content.parts if part.inline_data and part.inline_data.data),
                    None
                )
                if med_img_bytes:
                    med_img = Image.open(BytesIO(med_img_bytes))
                    buffered = BytesIO()
                    med_img.save(buffered, format="PNG")
                    med_img_str = base64.b64encode(buffered.getvalue()).decode()

                    st.markdown("<h4 style='text-align:center;'>Medicine Reference</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{med_img_str}' style='max-width:100%; border-radius:15px; box-shadow: 0 10px 20px rgba(0,0,0,0.2);'/></div>", unsafe_allow_html=True)
                else:
                    st.warning("No medicine image received.")
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
                response = client.models.generate_content(
                    model=TEXT_MODEL,
                    contents=[disease_prompt]
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error generating disease info: {e}")

        # Nutrition Image
        nutrition_prompt = f"Photorealistic nutrition plate for a person with {disease_name}, based on WHO/Mayo Clinic dietary guidance. Age: {age}, Gender: {gender}."
        with st.spinner("Generating nutrition image..."):
            try:
                img_response = client.models.generate_content(
                    model=IMAGE_MODEL,
                    contents=nutrition_prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                )
                img_bytes = next(
                    (part.inline_data.data for part in img_response.candidates[0].content.parts if part.inline_data and part.inline_data.data),
                    None
                )
                if img_bytes:
                    img = Image.open(BytesIO(img_bytes))
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    st.markdown("<h4 style='text-align:center;'>Nutrition Suggestion</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{img_str}' style='max-width:100%; border-radius:15px; box-shadow: 0 10px 20px rgba(0,0,0,0.2);'/></div>", unsafe_allow_html=True)
                else:
                    st.warning("No nutrition image received.")
            except Exception as e:
                st.error(f"Nutrition image error: {e}")

        # Medicine Image
        medicine_prompt = f"High-resolution image of common medicines or treatment kits for {disease_name}, recommended by WHO or Mayo Clinic. White background."
        with st.spinner("Generating medicine image..."):
            try:
                med_response = client.models.generate_content(
                    model=IMAGE_MODEL,
                    contents=medicine_prompt,
                    config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                )
                med_img_bytes = next(
                    (part.inline_data.data for part in med_response.candidates[0].content.parts if part.inline_data and part.inline_data.data),
                    None
                )
                if med_img_bytes:
                    med_img = Image.open(BytesIO(med_img_bytes))
                    buffered = BytesIO()
                    med_img.save(buffered, format="PNG")
                    med_img_str = base64.b64encode(buffered.getvalue()).decode()

                    st.markdown("<h4 style='text-align:center;'>Medicine Reference</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center;'><img src='data:image/png;base64,{med_img_str}' style='max-width:100%; border-radius:15px; box-shadow: 0 10px 20px rgba(0,0,0,0.2);'/></div>", unsafe_allow_html=True)
                else:
                    st.warning("No medicine image received.")
            except Exception as e:
                st.error(f"Medicine image error: {e}")

# Footer
st.markdown("<footer style='text-align:center; margin-top:3rem;'>Powered by Gemini 2.5 Flash & Gemini Image API</footer>", unsafe_allow_html=True)
