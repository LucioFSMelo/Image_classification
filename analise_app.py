import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zipfile
import os
import time
from io import BytesIO
from fpdf import FPDF
import tempfile
import random
from  PIL import Image

# Definindo do modelo CNN em PyTorch
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4, num_filters=32, filter_size=3, img_size=100):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=filter_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=filter_size, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_filters * (img_size // 2) * (img_size // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Função para carregar e preparar os dados de imagem
def load_image_data(path, _transform):
    dataset = datasets.ImageFolder(root=path, transform=_transform)
    return dataset

# Função para formatar o tempo em Horas:Minutos:Segundos
def format_elapsed_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return f"{hours:02}h:{minutes:02}m:{seconds:02}s"

# Função para gerar o relatório em PDF
def generate_pdf_report(metrics_df, accuracy_fig, loss_fig, elapsed_time, dict_inputs):
    pdf = FPDF()
    pdf.add_page()

    # Adicionando título
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Métricas do Modelo", ln=True, align="C")

    # Adicionando o dicionário dos inputs
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Inputs:", ln=True)
    for key, value in dict_inputs.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)


    # Adicionando tabela de métricas
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Métricas:", ln=True)
    for i in range(len(metrics_df)):
        row = metrics_df.iloc[i]
        pdf.cell(200, 10, txt=str(row), ln=True)


    # Calculando o tempo decorrido
    tempo_formatado = format_elapsed_time(elapsed_time)
    tempo = f"Tempo de Análise: {tempo_formatado}"
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=tempo, ln=True, align="C")

    # Adicionando gráficos
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Gráficos de Acurácia e Perda", ln=True, align="C")

    # Salvando as figuras em arquivos temporários
    accuracy_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    loss_img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    accuracy_fig.savefig(accuracy_img_temp.name, bbox_inches='tight')
    loss_fig.savefig(loss_img_temp.name, bbox_inches='tight')

    # Adicionando as imagens ao PDF
    pdf.image(accuracy_img_temp.name, x=10, y=20, w=90)
    pdf.image(loss_img_temp.name, x=110, y=20, w=90)

    # Removendo arquivos temporários de imagens
    os.remove(accuracy_img_temp.name)
    os.remove(loss_img_temp.name)

    # Salvando o PDF em um arquivo temporário
    temp_pdf_file = tempfile.NamedTemporaryFile(delete=False)
    temp_pdf_file.close()  # Fecha o arquivo para que o FPDF possa acessá-lo

    pdf.output(temp_pdf_file.name)

    # Lendo o conteúdo do arquivo temporário como bytes
    with open(temp_pdf_file.name, "rb") as f:
        pdf_bytes = f.read()

    # Removendo o arquivo temporário
    os.remove(temp_pdf_file.name)

    return pdf_bytes
# Função para mostrar uma imagem aleatória
def show_random_image(image_dir):
    image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('png', 'jpg', 'jpeg'))]
    if image_files:
        random_image = random.choice(image_files)
        image = Image.open(random_image)
        st.image(image, caption='Imagem Aleatória do Dataset', use_column_width=True)
    else:
        st.warning('Nenhuma imagem encontrada no diretório especificado.')

# Definindo os parâmetros e categorias do modelo
CATEGORIES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

# Variáveis globais
timer_start = 0
timer_running = False

# Título do aplicativo no Streamlit
st.title('Classificação de imagens usando CNN com PyTorch')
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

path_test = "dataset"

if st.button("Mostrar Imagem Aleatória"):
    # Mostrando uma imagem aleatória do dataset
    image_dir = "dataset/dataset-master/dataset-master/JPEGImages"
    show_random_image(image_dir)

# Inputs para hiperparâmetros e configuração da rede
img_size = st.number_input('Tamanho da Imagem', min_value=32, max_value=256, value=100)
num_filters = st.number_input('Número de Filtros', min_value=1, max_value=128, value=32)
filter_size = st.number_input('Tamanho dos Filtros', min_value=1, max_value=7, value=3)
learning_rate = st.number_input('Taxa de Aprendizado (Learning Rate)', min_value=0.0001, max_value=0.1, value=0.001, format="%.5f")
batch_size = st.number_input('Tamanho do Batch', min_value=1, max_value=128, value=16)
nb_epochs = st.number_input('Escolha o número de épocas', min_value=1, max_value=20, value=5)
dict_inputs = {"Tamanho da Imagem": img_size, "Núemro de Filtros": num_filters,
                "Tamanho dos Filtros": filter_size, "Learning Rate": learning_rate,
                "Tamanho do Batch": batch_size, "Número de Épocas": nb_epochs}

# Botão para iniciar a análise e o treinamento do modelo
if st.button('Iniciar Análise e Treinamento'):
    timer_start = time.time()
    timer_running = True
    if path_test:
        # Etapa 2: Preparando conjunto de dados para treinamento
        st.write("Preparando conjunto de dados")
        st.write("STATUS: Processando...")
        path_test = os.path.join(path_test, "dataset2-master", "dataset2-master", "images", "TRAIN")
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        dataset = load_image_data(path_test, transform)

        if not dataset:
            st.write("STATUS: ERRO")
            st.error("Erro ao carregar os dados. Verifique a estrutura do dataset.")
        else:
            # Dividindo o dataset em treino e teste
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Inicializando o modelo
            model = SimpleCNN(num_classes=4, num_filters=num_filters, filter_size=filter_size, img_size=img_size)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            st.markdown("STATUS: Concluido!")

            # Treinamento do modelo
            st.write("Treinando o Modelo")
            st.write("STATUS: Em andamento...")
            train_loss_history = []
            val_loss_history = []
            train_acc_history = []
            val_acc_history = []

            for epoch in range(nb_epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                train_loss = running_loss / len(train_loader)
                train_acc = 100. * correct / total
                train_loss_history.append(train_loss)
                train_acc_history.append(train_acc)

                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                val_loss /= len(test_loader)
                val_acc = 100. * correct / total
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)

                st.write(f'Época {epoch + 1}/{nb_epochs}, Perda de Treinamento: {train_loss:.4f}, Acurácia de Treinamento: {train_acc:.2f}%, Perda de Validação: {val_loss:.4f}, Acurácia de Validação: {val_acc:.2f}%')

            # Visualizando os resultados
            st.write("Treinamento Concluído")
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].plot(train_loss_history, label='Treinamento')
            ax[0].plot(val_loss_history, label='Validação')
            ax[0].set_title('Perda')
            ax[0].set_xlabel('Épocas')
            ax[0].set_ylabel('Perda')
            ax[0].legend()
            st.pyplot(fig)

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[1].plot(train_acc_history, label='Treinamento')
            ax[1].plot(val_acc_history, label='Validação')
            ax[1].set_title('Acurácia')
            ax[1].set_xlabel('Épocas')
            ax[1].set_ylabel('Acurácia')
            ax[1].legend()
            st.pyplot(fig)

            # Gerando relatório em PDF
            metrics_data = {
                'Época': list(range(1, nb_epochs + 1)),
                'Perda de Treinamento': train_loss_history,
                'Acurácia de Treinamento': train_acc_history,
                'Perda de Validação': val_loss_history,
                'Acurácia de Validação': val_acc_history,
            }

            # Calculando o tempo decorrido
            timer_end = time.time()
            elapsed_time = timer_end - timer_start
            tempo_formatado = format_elapsed_time(elapsed_time)
            st.write(f"Tempo de Análise: {tempo_formatado}")

            metrics_df = pd.DataFrame(metrics_data)
            pdf_report = generate_pdf_report(metrics_df, fig, fig, elapsed_time, dict_inputs)
            st.download_button("Download do Relatório PDF", data=pdf_report, file_name="relatorio_metrica.pdf", mime="application/pdf")

