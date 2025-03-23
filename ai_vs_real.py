import streamlit as st
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import numpy as np

model_name = "prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2"

@st.cache_resource(show_spinner=False)
def load_model():
    model = SiglipForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

st.title("AuthenSight")
st.write("Upload an image to determine whether it is synthetic (AI-generated or deepfake) or a real image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def image_classification(image: Image.Image):
    """
    Classifies an image as Synthetic (AI/Deepfake) or Real.
    """
   
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
 
    labels = model.config.id2label 
    predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}
    
 
    synthetic_prob = predictions.get("AI", 0) + predictions.get("Deepfake", 0)
    real_prob = predictions.get("Real", 0)
    
    combined = {
        "Synthetic (AI/Deepfake)": round(synthetic_prob, 3),
        "Real": round(real_prob, 3)
    }
    
    return combined

if uploaded_file is not None:
  
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
 
    combined_predictions = image_classification(image)
    st.write("### Classification Result")
    for label, prob in combined_predictions.items():
        st.write(f"**{label}**: {prob}")
