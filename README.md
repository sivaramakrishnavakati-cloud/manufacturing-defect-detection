# AI-Powered Manufacturing Defect Detection

## 🚀 Live Demo

Try the deployed application here:

[Launch Manufacturing Defect Detection App](https://sivaramakrishnavakati-cloud-manufacturing-defect--srcapp-yhgpb7.streamlit.app)

## Features

- Steel surface defect classification using ResNet18
- Six defect categories
- Prediction confidence and probability visualization
- Downloadable CSV prediction reports
- PyTorch-based deep learning model
- Streamlit web application
- Private model hosting using Hugging Face

## Disclaimer

This project is an experimental portfolio demonstration.
Predictions should not be used as the sole basis for industrial
quality-control decisions. Independent testing on representative
production data is required before real-world use.

A deep-learning application that classifies steel surface images into six defect categories using PyTorch and a pretrained ResNet18 model.

## Project Overview

Manufacturing quality inspection can benefit from computer vision systems that help identify potential surface defects.

This project demonstrates an image-classification workflow that includes dataset preparation, model training, evaluation, and an interactive prediction application.

## Features

- Classifies steel surface images into six categories.
- Uses transfer learning with ResNet18.
- Displays the predicted defect and model confidence.
- Visualizes probabilities for all six categories.
- Exports individual prediction results to CSV.
- Includes evaluation scripts and a confusion matrix.

## Defect Categories

1. Crazing
2. Inclusion
3. Patches
4. Pitted surface
5. Rolled-in scale
6. Scratches

## Technologies

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow

## Model Results

The initial CNN achieved 58.06% validation accuracy.

The ResNet18 transfer-learning experiment achieved a best validation accuracy of 99.17% on 360 validation images.

The final evaluation script reported 99.17% accuracy on that same validation set.

**Important:** The validation set was used during model selection. These results should not be interpreted as independent test performance or as a guarantee of accuracy on real factory images. Further evaluation on a separate, representative test dataset is required.

## Project Structure

```text
manufacturing-defect-detection/
├── src/
│   ├── app.py
│   ├── check_environment.py
│   ├── evaluate_model.py
│   ├── evaluate_resnet18.py
│   ├── inspect_dataset.py
│   ├── train_model.py
│   └── train_resnet18.py
├── models/
├── results/
├── data/
├── .gitignore
├── README.md
└── requirements.txt
