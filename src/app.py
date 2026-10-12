from pathlib import Path

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "defect_resnet18.pth"

HF_REPO_ID = "sivaramakrishnav/manufacturing-defect-detection-model"
MODEL_FILENAME = "defect_resnet18.pth"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224


# --------------------------------------------------
# 2. Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="AI Manufacturing Defect Detection",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 AI-Powered Manufacturing Defect Detection")

st.write(
    "Upload a steel surface image to predict its defect category "
    "using a ResNet18 deep-learning model."
)

st.info(
    "This is an experimental portfolio project. Predictions should "
    "not be used as the sole basis for industrial quality-control decisions."
)


# --------------------------------------------------
# 3. Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 4. Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    if LOCAL_MODEL_PATH.exists():
        model_path = str(LOCAL_MODEL_PATH)
    else:
        hf_token = st.secrets.get("HF_TOKEN", None)

        if not hf_token:
            raise RuntimeError(
                "Model access token is missing. "
                "Configure HF_TOKEN in Streamlit secrets."
            )

        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            token=hf_token,
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )
    class_names = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


# --------------------------------------------------
# 5. Upload image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Choose a clear image of a steel surface.",
)


# --------------------------------------------------
# 6. Predict defect
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        left_column, right_column = st.columns(2)

        with left_column:
            st.subheader("Uploaded Image")
            st.image(
                image,
                caption=uploaded_file.name,
                width="stretch",
            )

        with st.spinner("Analyzing the image..."):

            model, class_names = load_model()

            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor.to(DEVICE))
                probabilities = torch.softmax(outputs, dim=1)[0]

            predicted_index = int(probabilities.argmax().item())
            predicted_class = class_names[predicted_index]
            confidence = float(probabilities[predicted_index].item())

        with right_column:
            st.subheader("Prediction Results")

            st.metric(
                label="Predicted Defect",
                value=predicted_class.replace("_", " ").title(),
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2%}",
            )

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

        st.divider()

        st.subheader("Probabilities for All Defect Categories")

        probability_data = {
            class_name.replace("_", " ").title(): float(probability)
            for class_name, probability in zip(
                class_names,
                probabilities.tolist(),
            )
        }

        st.bar_chart(probability_data)
                # Download the prediction as a CSV report
        report = pd.DataFrame([
            {
                "image_name": uploaded_file.name,
                "predicted_defect": predicted_class,
                "confidence_percent": round(confidence * 100, 2),
                **{
                    f"{name}_probability_percent": round(
                        float(probability) * 100, 2
                    )
                    for name, probability in zip(
                        class_names,
                        probabilities.tolist(),
                    )
                },
            }
        ])

        st.download_button(
            label="Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="defect_prediction_report.csv",
            mime="text/csv",
        )

        st.subheader("Interpretation")

        if confidence >= 0.80:
            st.write(
                f"The model most strongly predicts **{predicted_class.replace('_', ' ').title()}**."
            )
        else:
            st.warning(
                "The model's highest prediction probability is below 80%. "
                "Review the image and verify the result independently."
            )

        st.caption(
            "The 80% threshold is an illustrative interface setting, "
            "not a validated industrial acceptance threshold."
        )

    except Exception as error:
        st.error(f"Unable to process this image: {error}")


# --------------------------------------------------
# 7. Sidebar information
# --------------------------------------------------

with st.sidebar:
    st.header("About This Project")

    st.write("**Model:** ResNet18")
    st.write("**Framework:** PyTorch")
    st.write("**Task:** Steel surface defect classification")

    st.write("### Supported categories")

    for defect in [
        "Crazing",
        "Inclusion",
        "Patches",
        "Pitted surface",
        "Rolled-in scale",
        "Scratches",
    ]:
        st.write(f"- {defect}")

    st.divider()

    st.write(
        "Developed as a machine-learning portfolio project."
    )