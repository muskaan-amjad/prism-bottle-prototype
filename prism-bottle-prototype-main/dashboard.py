import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os

# === Config paths ===
MODEL_PATH = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/models/best_model.h5"
VAL_LABELS_CSV = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val_labels.csv"
VAL_IMAGES_DIR = "/Users/muskanamjad/Downloads/prism-bottle-prototype-main/prism-bottle-prototype-main/data/val"

# === LOAD MODEL & LABELS ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

@st.cache_data
def load_val_labels():
    df = pd.read_csv(VAL_LABELS_CSV)
    # Normalize filenames in CSV index: strip whitespace and lowercase
    df['image_filename'] = df['image_filename'].str.strip().str.lower()
    df.set_index('image_filename', inplace=True)
    return df

model = load_model()
val_labels = load_val_labels()

def feedback(score):
    if score >= 8:
        return "Excellent"
    elif score >= 6:
        return "Good"
    elif score >= 4:
        return "Average"
    else:
        return "Needs Improvement"

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    arr = np.array(img_resized).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return image, arr

def predict_scores(arr):
    pred = model.predict(arr)
    return pred[0]

def display_comparison(image_name, pred_scores):
    normalized_name = image_name.strip().lower()
    if normalized_name in val_labels.index:
        true_scores = val_labels.loc[normalized_name]
        # Explicit float conversion to avoid pandas series/potential dtype issues
        actual = [
            float(true_scores['durability_score']),
            float(true_scores['chemical_safety_score']),
            float(true_scores['ergonomics_score'])
        ]
    else:
        actual = ["N/A"] * 3
        st.warning(f"No ground truth available for {image_name}")

    data = {
        "Metric": ["Durability", "Chemical Safety", "Ergonomics"],
        "Predicted Score": [round(float(x), 2) for x in pred_scores],
        "AI Feedback": [feedback(float(x)) for x in pred_scores],
        "Actual Score": actual
    }
    df = pd.DataFrame(data)
    st.table(df)

# === SIDEBAR NAVIGATION ===
st.sidebar.markdown("### Navigation")
menu = st.sidebar.radio(
    "Go to",
    [
        "Upload & Predict",
        "Compare Bottles",
        "ChatGPT Assistant",
        "Feedback Form"
    ]
)

# === MAIN HEADER & DESCRIPTION ===
if menu == "Upload & Predict":
    st.markdown(
        "<h1 style='text-align: center; color: #e60073;'>AI Powered Product Analysis Dashboard</h1>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='text-align: center; font-size:1.1em'>
    <b>Upload a <span style='color:#1976D2'>bottle image</span></b> and receive predictions for:
    <ul style='list-style:disc; margin-left:2em; text-align:left;'>
      <li>Master Category</li>
      <li>Subtype</li>
      <li>Morphological Features</li>
      <li>Functional Factors</li>
      <li>Real World Usage Traits</li>
    </ul>
    <i>Use the integrated <b>AI assistant</b> for expert insights, and download a PDF report.</i>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload bottle image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        filename_norm = uploaded_file.name.strip().lower()
        image, arr = preprocess_image(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        pred_scores = predict_scores(arr)
        display_comparison(filename_norm, pred_scores)

elif menu == "Compare Bottles":
    st.markdown("<h2>Compare Bottles</h2>", unsafe_allow_html=True)
    image_files = [f for f in os.listdir(VAL_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    img1 = st.selectbox("Select First Bottle Image", image_files)
    img2 = st.selectbox("Select Second Bottle Image", image_files, index=1 if len(image_files) > 1 else 0)

    col1, col2 = st.columns(2)

    def load_and_predict(image_name):
        # Map normalized name back to actual filename (case insensitive)
        matched_file = next((f for f in image_files if f.strip().lower() == image_name.strip().lower()), None)
        if matched_file is None:
            st.error(f"Cannot find image file {image_name}")
            return None, [0, 0, 0]

        image_path = os.path.join(VAL_IMAGES_DIR, matched_file)
        image = Image.open(image_path).convert("RGB")
        img_resized = image.resize((224, 224))
        arr = np.array(img_resized).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = model.predict(arr)
        return image, pred[0]

    with col1:
        st.subheader("First Bottle")
        img1_obj, pred1 = load_and_predict(img1)
        if img1_obj:
            st.image(img1_obj, caption=img1, use_container_width=True)
            display_comparison(img1.lower(), pred1)

    with col2:
        st.subheader("Second Bottle")
        img2_obj, pred2 = load_and_predict(img2)
        if img2_obj:
            st.image(img2_obj, caption=img2, use_container_width=True)
            display_comparison(img2.lower(), pred2)

elif menu == "ChatGPT Assistant":
    st.markdown("### ChatGPT Assistant")
    st.info("Coming soon: Integrated AI assistant for insights and automated PDF reports.")

elif menu == "Feedback Form":
    st.markdown("<h2>Give Us Feedback</h2>", unsafe_allow_html=True)
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        comments = st.text_area("Comments or Suggestions")
        submitted = st.form_submit_button("Submit")

        if submitted:
            st.success("Thank you for your feedback!")
