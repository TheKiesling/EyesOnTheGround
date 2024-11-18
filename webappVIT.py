import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from joblib import load

# Cargar el modelo híbrido
vit_model = load('models/vitModel2.pkl')
column_transformer_vit = load("models/column_transformerVIT.pkl")

rf_model = load("CNN_RF_with_augmentation_model/random_forest_aug_model.pkl")  # Modelo Random Forest
vgg16_model = tf.keras.models.load_model("CNN_RF_with_augmentation_model/model_vgg16.keras")  # Modelo VGG16
column_transformer_rf = load("CNN_RF_with_augmentation_model/column_transformer.pkl")

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
    
    # Selección del modelo a utilizar
    model_choice = st.radio(
        "Selecciona el enfoque a utilizar:",
        ('Vision Transformers (ViT) + Regresión Densa', 'CNN + Random Forest')
    )

    if model_choice == 'Vision Transformers (ViT) + Regresión Densa':
        # Aplicar el ColumnTransformer para el modelo híbrido
        encoded_features_vit = column_transformer_vit.transform(input_features)

        # Eliminar columnas innecesarias si es necesario
        encoded_features_dataset = pd.DataFrame(encoded_features_vit, columns=column_transformer_vit.get_feature_names_out())
        encoded_features_dataset = encoded_features_dataset.drop(columns=['remainder__ID', 'remainder__filename', 'remainder__extent'], errors='ignore')
        encoded_features_vit = encoded_features_dataset.to_numpy()

        # Preprocesar la imagen
        image_tensor = process_image(image)

        # Predicción usando el modelo híbrido
        predictions = vit_model.predict({
            'image_input': image_tensor,
            'tabular_input': encoded_features_vit.astype(np.float32)
        })

        st.write(f"Predicción del nivel de extensión de la enfermedad (usando Vision Transformers): {predictions[0][0]:.2f}")

    elif model_choice == 'CNN + Random Forest':
        # Aplicar el ColumnTransformer para el modelo Random Forest
        encoded_features_rf = column_transformer_rf.transform(input_features)

        # Eliminar columnas innecesarias si es necesario
        encoded_features_rf_dataset = pd.DataFrame(encoded_features_rf, columns=column_transformer_rf.get_feature_names_out())
        encoded_features_rf_dataset = encoded_features_rf_dataset.drop(columns=['remainder__ID', 'remainder__filename', 'remainder__extent'], errors='ignore')
        encoded_features_rf = encoded_features_rf_dataset.to_numpy()

        # Preprocesar la imagen y extraer características usando el modelo VGG16
        image_tensor = process_image(image)
        features = vgg16_model.predict(image_tensor)
        features_flat = features.reshape(1, -1)

        # Combinar características
        combined_features_rf = np.hstack((features_flat, encoded_features_rf))

        # Predicción usando Random Forest
        predictions_rf = rf_model.predict(combined_features_rf)
        st.write(f"Predicción del nivel de extensión de la enfermedad (usando CNN + Random Forest): {predictions_rf[0]:.2f}")

else:
    st.write('Por favor, sube una imagen para continuar.')
