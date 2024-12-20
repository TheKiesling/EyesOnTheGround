import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from joblib import load
import matplotlib.pyplot as plt



# Cargar el modelo híbrido
vit_model = load('models/vitModel2.pkl')
column_transformer_vit = load("models/column_transformerVIT.pkl")

rf_model = load("models/random_forest_aug_model.pkl")  # Modelo Random Forest
vgg16_model = tf.keras.models.load_model("models/model_vgg16.keras")  # Modelo VGG16
column_transformer_rf = load("models/column_transformer.pkl")

IMAGE_SIZE = 224

# Función para procesar la imagen
def process_image(image, size):
    img = image.resize((size, size))
    img = np.array(img) / 255.0  # Normalización
    img = np.expand_dims(img, axis=0)
    return img

def get_attention_map(model, image):
    
    # Preprocesar la imagen
    image = tf.image.resize(image, (224, 224))
    image = tf.expand_dims(image, axis=0)
    image = image / 255.0

    # Preprocesar las características tabulares
    feature_columns = ['growth_stage_F', 'growth_stage_M', 'growth_stage_S', 'growth_stage_V',
        'damage_DR', 'damage_DS', 'damage_FD', 'damage_G', 'damage_ND',
        'damage_PS', 'damage_WD', 'damage_WN', 'season_LR2020', 'season_LR2021',
        'season_SR2020', 'season_SR2021']
    
    dummy_tabular_input = np.zeros((1, len(feature_columns)), dtype=np.float32)

    # Crear el diccionario de entradas
    inputs = {
        'image_input': image,
        'tabular_input': dummy_tabular_input
    }

    # Encontrar las capas MultiHeadAttention
    attention_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MultiHeadAttention)]

    if len(attention_layers) == 0:
        raise ValueError("No se encontraron capas MultiHeadAttention en el modelo.")

    # Crear un modelo intermedio para obtener las activaciones de la primera capa de atención
    first_attention_layer = attention_layers[0]
    intermediate_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=first_attention_layer.output  # Activaciones de la capa de atención
    )

    # Obtener las activaciones de la capa de atención
    attention_output = intermediate_model.predict(inputs)
    
    # La salida de la capa de atención es [batch_size, num_heads, seq_len, seq_len], promediamos por los heads y la secuencia
    if len(attention_output.shape) == 4:
        # Promediar sobre la dimensión de los heads
        attention_map = np.mean(attention_output, axis=1)  # Promediar sobre los heads
        attention_map = np.mean(attention_map, axis=2)  # Promediar sobre la secuencia
    else:
        attention_map = attention_output

    return attention_map
    
def plot_attention_map(model, image):
    # Convertir la imagen original
    original_image = tf.keras.preprocessing.image.array_to_img(image)

    # Obtener el mapa de atención
    attention_map = get_attention_map(model, image)

    # Redimensionar el mapa de atención para que coincida con el tamaño de la imagen
    attention_map_resized = tf.image.resize(
        tf.expand_dims(attention_map, axis=-1), 
        (image.shape[0], image.shape[1])
    ).numpy().squeeze()

    # Graficar
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(original_image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Mapa de Atención")
    plt.imshow(attention_map_resized, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()

st.title('Detección y Predicción de Enfermedades en Plantas - Solución Híbrida (Vision Transformers)')

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
        image_tensor = process_image(image, 224)

        # Predicción usando el modelo híbrido
        predictions = vit_model.predict({
            'image_input': image_tensor,
            'tabular_input': encoded_features_vit.astype(np.float32)
        })

        st.write(f"Predicción del nivel de extensión de la enfermedad (usando Vision Transformers): {predictions[0][0]:.2f}")
        
        st.write("Mapa de atención en análisis de imagen:")
        
        # Mostrar el mapa de atención
        # Visualizar el mapa de atención
        img = tf.keras.preprocessing.image.img_to_array(image)
        
        plot_attention_map(vit_model, img)
        
        # Mostrar el gráfico en Streamlit
        st.pyplot(plt)

    elif model_choice == 'CNN + Random Forest':
        # Aplicar el ColumnTransformer para el modelo Random Forest
        encoded_features_rf = column_transformer_rf.transform(input_features)

        # Eliminar columnas innecesarias si es necesario
        encoded_features_rf_dataset = pd.DataFrame(encoded_features_rf, columns=column_transformer_rf.get_feature_names_out())
        encoded_features_rf_dataset = encoded_features_rf_dataset.drop(columns=['remainder__ID', 'remainder__filename', 'remainder__extent'], errors='ignore')
        encoded_features_rf = encoded_features_rf_dataset.to_numpy()

        # Preprocesar la imagen y extraer características usando el modelo VGG16
        image_tensor = process_image(image,128)
        features = vgg16_model.predict(image_tensor)
        features_flat = features.reshape(1, -1)

        # Combinar características
        combined_features_rf = np.hstack((features_flat, encoded_features_rf))

        # Predicción usando Random Forest
        predictions_rf = rf_model.predict(combined_features_rf)
        st.write(f"Predicción del nivel de extensión de la enfermedad (usando CNN + Random Forest): {predictions_rf[0]:.2f}")
        
        st.write("Importancia de las características:")
        
        # Obtener la importancia de cada característica
        importances = rf_model.feature_importances_
        features = np.hstack((encoded_features_rf_dataset.columns, ['imagen'] * features_flat.shape[1]))

        # Crear un DataFrame con la importancia
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Agrupar las características con nombres numéricos bajo la categoría "imagen"
        importance_df['Feature'] = importance_df['Feature'].apply(lambda x: 'imagen' if x.isdigit() else x)

        # Sumar las importancias de todas las columnas categorizadas como "imagen"
        grouped_importances = (
            importance_df.groupby('Feature')['Importance'].sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        # Limitar a las N categorías más importantes
        N = 10  # Ajustar según sea necesario
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
