import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt

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

st.title('Detección y Predicción de Enfermedades en Plantas')
st.write('Sube una imagen de la planta para predecir si está enferma y el nivel de extensión de la enfermedad.')

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

    # Asegurar que las columnas esperadas están presentes (aunque vacías)
    expected_columns = ['extent', 'filename', 'ID', 'growth_stage', 'damage', 'season']
    for col in expected_columns:
        if col not in categorical_features.columns:
            categorical_features[col] = None

    one_hot_encoder = pickle.load(open('models/one_hot_encoder.pkl', 'rb'))  # Asegúrate de que tu codificador está entrenado correctamente
    encoded_features = one_hot_encoder.transform(categorical_features)

    # Asegurar que encoded_features es un arreglo plano si es necesario
    if encoded_features.ndim == 2:
        encoded_features = encoded_features.flatten()

    # Transformar imagen a tensor y extraer características
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = resnet50(image_tensor).cpu().numpy().flatten()

    # Combinar las características
    combined_features = np.hstack((image_features, encoded_features))

    # Ajustar el tamaño de las características si es necesario
    expected_feature_count = 2064  # Cambiar a la cantidad que tu modelo espera
    if combined_features.size > expected_feature_count:
        combined_features = combined_features[:expected_feature_count]
    elif combined_features.size < expected_feature_count:
        combined_features = np.pad(combined_features, (0, expected_feature_count - combined_features.size), mode='constant')

    # Predicción de la enfermedad
    is_diseased = xgb_classifier.predict([combined_features])[0]
    st.write(f"Predicción de enfermedad: {'Enferma' if is_diseased else 'Sana'}")

    if is_diseased:
        disease_extent = rf_regressor.predict([combined_features])[0]
        st.write(f"Predicción del nivel de extensión de la enfermedad: {disease_extent:.2f}")
        
    importances = rf_regressor.feature_importances_
        
    num_importances = len(importances)
    features = np.hstack((np.arange(image_features.size), one_hot_encoder.get_feature_names_out()))

    if len(features) > num_importances:
        features = features[:num_importances]
    elif len(features) < num_importances:
        importances = importances[:len(features)]

    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

    importance_df['Feature'] = importance_df['Feature'].apply(lambda x: 'imagen' if str(x).isdigit() else x)

    grouped_importances = (
        importance_df.groupby('Feature')['Importance'].sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    N = 10
    top_features = grouped_importances.iloc[:N]

    # Graficar la importancia de cada categoría
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'], top_features['Importance'], color='#4e6b4d')
    plt.xlabel('Importancia')
    plt.ylabel('Categorías')
    plt.title(f'Importancia de las Top {N} categorías en el Random Forest')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().invert_yaxis()

    # Mostrar el gráfico en Streamlit
    st.pyplot(plt)
    
else:
    st.write('Por favor, sube una imagen para continuar.')