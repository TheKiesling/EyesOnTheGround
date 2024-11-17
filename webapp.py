import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
import joblib
from tensorflow.keras.models import load_model

st.title('Detección y Predicción de Enfermedades en Plantas')
st.write('Selecciona el modelo que deseas usar:')

model_option = st.radio(
    "Modelos disponibles:",
    ('Modelo ResNet50 y XGBoost', 'Modelo Random Forest')
)

if model_option == 'Modelo ResNet50 y XGBoost':
    with open('models/xgb_classifier.pkl', 'rb') as file:
        xgb_classifier = pickle.load(file)

    with open('models/rf_regressor.pkl', 'rb') as file:
        rf_regressor = pickle.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    data_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    resnet50 = resnet50.to(device)
    resnet50.eval()
elif model_option == 'Modelo Random Forest':
    column_transformer = joblib.load("models/column_transformer.pkl")
    rf_regressor = joblib.load("models/random_forest_aug_model.pkl")

    model = load_model("models/model_vgg16.keras")

uploaded_file = st.file_uploader("Sube una imagen de la planta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_container_width=True)
    st.write('Por favor, ingresa la siguiente información:')

    growth_stage = ['F', 'M', 'S', 'V']
    damage = ['DR', 'DS', 'FD', 'G', 'ND', 'PS', 'WD', 'WN']
    season = ['SR2020', 'SR2021', 'LR2020', 'LR2021']

    selected_growth_stage = st.selectbox('Etapa de crecimiento', growth_stage)
    selected_damage = st.selectbox('Daño', damage)
    selected_season = st.selectbox('Temporada', season)

    categorical_features = pd.DataFrame({
        'growth_stage': [selected_growth_stage],
        'damage': [selected_damage],
        'season': [selected_season]
    })

    if model_option == 'Modelo ResNet50 y XGBoost':
        one_hot_encoder = pickle.load(open('models/one_hot_encoder.pkl', 'rb'))
        encoded_features = one_hot_encoder.transform(categorical_features)
        encoded_features = encoded_features.flatten() if encoded_features.ndim == 2 else encoded_features

        image_tensor = data_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = resnet50(image_tensor).cpu().numpy().flatten()

        combined_features = np.hstack((image_features, encoded_features))
        expected_feature_count = 2064 
        if combined_features.size > expected_feature_count:
            combined_features = combined_features[:expected_feature_count]
        elif combined_features.size < expected_feature_count:
            combined_features = np.pad(combined_features, (0, expected_feature_count - combined_features.size), mode='constant')

        is_diseased = xgb_classifier.predict([combined_features])[0]
        st.write(f"Predicción de enfermedad: {'Enferma' if is_diseased else 'Sana'}")

        if is_diseased:
            disease_extent = rf_regressor.predict([combined_features])[0]
            st.write(f"Predicción del nivel de extensión de la enfermedad: {disease_extent:.2f}")

    elif model_option == 'Modelo Random Forest':
        encoded_features = column_transformer.transform(categorical_features)
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        prediction = model.predict(image)
        prediction = np.argmax(prediction, axis=1)
        prediction = prediction[0]
        prediction = rf_regressor.predict(np.concatenate((encoded_features, [[prediction]]), axis=1))
        
        st.write(f"Predicción del nivel de extensión de la enfermedad: {prediction[0]:.2f}")
       

else:
    st.write('Por favor, sube una imagen para continuar.')
