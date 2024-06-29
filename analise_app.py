import zipfile

import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Função para carregar e preparar os dados de imagem
def load_image_data(path, categories, img_size):
    data = []
    for category in categories:
        category_path = os.path.join(path, category)
        class_num = categories.index(category)
        for img in os.listdir(category_path):
            try:
                img_array = cv2.imread(os.path.join(category_path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                data.append([new_array, class_num])
            except Exception as e:
                pass
    return data

# Definindo os parâmetros e categorias do modelo
# C:\Users\luciu\Workspace\Senac_proj\Computer_Vision\uploaded_dataset\dataset2-master\dataset2-master\images
CATEGORIES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = 200
path_test = ""

# Título do aplicativo no Streamlit
st.title('Classificação de imagens usando CNN com Keras e CIFAR-10')
st.markdown("""
**Disciplina: Visão Computacional**  
**Professor:** Alex Cordeiro  
**Alunos:**   
    João Pedro 
    José Victor   
    Lucio Flavio  
    Néliton Vanderley  
    Wellington França  
""")
st.markdown("""
Este projeto foi desenvolvido com base em técnicas descritas no artigo "[Image Classification Using CNN with Keras & CIFAR-10](https://www.analyticsvidhya.com/blog/2021/01/image-classification-using-convolutional-neural-networks-a-step-by-step-guide/)",
 que fornece uma visão abrangente sobre a aplicação de redes neurais convolucionais na
  classificação de células sanguíneas, destacando a importância da automação na análise de 
  imagens médicas para melhorar a precisão e a eficiência dos diagnósticos laboratoriais.
""")
st.markdown("[Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)")
st.markdown("Acesse nosso GitHub do projeto [aqui](https://github.com/LucioFSMelo/Cnn_classification).")

# Etapa 1: Upload do dataset
uploaded_files = st.file_uploader("Faça o upload de um arquivo zip com o dataset", type="zip")

if uploaded_files is not None:
    with open("uploaded_dataset.zip", "wb") as f:
        f.write(uploaded_files.getbuffer())
    st.success("Upload concluído!")
    path_test = "uploaded_dataset"
    with zipfile.ZipFile("uploaded_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(path_test)
    st.success("Dataset extraído com sucesso!")

# Input para escolher o número de épocas
nb_epochs = st.number_input('Escolha o número de épocas', min_value=1, max_value=20, value=5)

# Botão para iniciar a análise e o treinamento do modelo
if st.button('Iniciar Análise e Treinamento'):
    if path_test:
        # Etapa 2: Preparar conjunto de dados para treinamento
        st.write("Carregando e preparando os dados...")
        path_test = path_test + "/dataset2-master/dataset2-master/images/TRAIN"
        training_data = load_image_data(path_test, CATEGORIES, IMG_SIZE)

        # Etapa 3: Embaralhar o conjunto de dados
        random.shuffle(training_data)

        # Etapa 4: Atribuindo rótulos e recursos
        X = []
        y = []
        for features, label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        y = np.array(y)

        # Etapa 5: Normalizando X e convertendo rótulos em dados categóricos
        X = X.astype('float32') / 255.0
        y = tf.keras.utils.to_categorical(y, 4)

        # Etapa 6: Dividir X e Y para uso na CNN
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        # Etapa 7: Definir, compilar e treinar o modelo CNN
        batch_size = 16
        nb_classes = 4

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(nb_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Treinamento do modelo
        st.write("Treinando o modelo...")
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test))

        # Etapa 8: Precisão e pontuação do modelo
        score = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Score: {score[0]}")
        st.write(f"Test Accuracy: {score[1]}")

        # Plotar a precisão e a perda
        st.write("Visualização da Acurácia e Perda")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['accuracy'], label='Train Accuracy')
        ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax[0].legend()
        ax[0].set_title('Accuracy')
        ax[1].plot(history.history['loss'], label='Train Loss')
        ax[1].plot(history.history['val_loss'], label='Validation Loss')
        ax[1].legend()
        ax[1].set_title('Loss')
        st.pyplot(fig)

        # Gerar relatório
        report = {
            "Test Score": score[0],
            "Test Accuracy": score[1],
            "Training History": history.history
        }
        report_df = pd.DataFrame(report)
        report_path = "model_report.csv"
        report_df.to_csv(report_path)
        st.write("Relatório gerado com sucesso!")
        st.download_button(label="Download Relatório", data=open(report_path, "rb").read(), file_name=report_path)

    else:
        st.error("Por favor, faça o upload do dataset antes de iniciar a análise e o treinamento.")
