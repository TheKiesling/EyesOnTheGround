import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
from joblib import load

# Cargar el modelo híbrido
vit_model = load('models/vitModel2.pkl')
column_transformer = load("models/column_transformerVIT.pkl")

IMAGE_SIZE = 224

# Función para procesar la imagen
def process_image(image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img) / 255.0  # Normalización
    img = np.expand_dims(img, axis=0)
    return img

st.title('Detección y Predicción de Enfermedades en Plantas - Solución Híbrida (Vision Transformers)')

uploaded_file = st.file_uploader("Sube una imagen de la planta", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_column_width=True)

    st.write('Por favor, ingresa la siguiente información:')
    
    growth_stage = ['F', 'M', 'S', 'V']
    damage = ['DR', 'DS', 'FD', 'G', 'ND', 'PS', 'WD', 'WN']
    season = ['SR2020', 'SR2021', 'LR2020', 'LR2021']
    
    selected_growth_stage = st.selectbox('Etapa de crecimiento', growth_stage)
    selected_damage = st.selectbox('Daño', damage)
    selected_season = st.selectbox('Temporada', season)

    # Crear DataFrame para características tabulares
    input_features = pd.DataFrame({
        'growth_stage': [selected_growth_stage],
        'damage': [selected_damage],
        'season': [selected_season],
        'ID': [None],
        'extent': [None],
        'filename': [None]
    })
    
    
    # Aplicar el ColumnTransformer para codificación
    encoded_features = column_transformer.transform(input_features)
    
    # Eliminar columnas innecesarias
    encoded_features_dataset = pd.DataFrame(encoded_features, columns=column_transformer.get_feature_names_out())
    encoded_features_dataset = encoded_features_dataset.drop(columns=['remainder__ID', 'remainder__filename', 'remainder__extent'])
    encoded_features = encoded_features_dataset.to_numpy()

    # Preprocesar la imagen
    image_tensor = process_image(image)

    # Predicción usando el modelo híbrido
    predictions = vit_model.predict({
        'image_input': image_tensor,
        'tabular_input': encoded_features.astype(np.float32)
    })

    st.write(f"Predicción del nivel de extensión de la enfermedad (usando Vision Transformers): {predictions[0][0]:.2f}")
else:
    st.write('Por favor, sube una imagen para continuar.')