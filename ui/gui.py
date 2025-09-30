# simple_gui.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import altair as alt

from processor import preprocess
from predictor import load_model, predict

st.set_page_config(page_title="Digit Demo", page_icon="🔢", layout="centered")
st.title("🖌️ Draw a digit (0–9)")

# Sidebar: weights file input

weights_file = st.sidebar.text_input("Weights file", value="model_digit2.pth")
model = load_model(weights_file)

# Drawing canvas

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=8,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280, height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Convert canvas to PIL image

img = None
if canvas and canvas.image_data is not None:
    arr = (canvas.image_data[:, :, :3] * 255).astype("uint8")
    img = Image.fromarray(arr)

# Prediction logic

if img is None:
    st.info("Draw a digit.")
else:
    x = preprocess(img)
    if x is None:
        st.warning("The drawing is too faint/empty. Try using a thicker brush and draw again.")
    else:
        pred, probs = predict(model, x)
        confidence = max(probs)  # Confidence = highest probability
        st.subheader(f"Prediction: **{pred}** (Confidence: {confidence:.2f})")

        # For visualization, we can still show all class probabilities
        df = pd.DataFrame({"digit": list(range(10)), "confidence": probs})

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("digit:O", title="Class (0–9)"),
                y=alt.Y("confidence:Q", title="Confidence", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(
                    alt.datum.digit == pred,
                    alt.value("#4C78A8"),
                    alt.value("#B3C3D3")
                ),
                tooltip=[
                    alt.Tooltip("digit:O", title="Class"),
                    alt.Tooltip("confidence:Q", title="Confidence", format=".3f")
                ],
            ).properties(height=260)
        )

        labels = (
            alt.Chart(df)
            .mark_text(dy=-6)
            .encode(
                x="digit:O",
                y=alt.Y("confidence:Q", scale=alt.Scale(domain=[0, 1])),
                text=alt.Text("confidence:Q", format=".2f"),
                color=alt.value("#333")
            )
        )

        st.altair_chart(chart + labels, use_container_width=True)
        st.caption("Note: Confidence indicates how sure the model is about its prediction.")
