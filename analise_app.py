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


# Definição do modelo CNN em PyTorch
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 50 * 50, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 32 * 50 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Função para carregar e preparar os dados de imagem
def load_image_data(path, _transform):
    dataset = datasets.ImageFolder(root=path, transform=_transform)
    return dataset


# Função para gerar o relatório em PDF
def generate_pdf_report(metrics_df, accuracy_fig, loss_fig):
    pdf = FPDF()
    pdf.add_page()

    # Adicionar título
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Métricas do Modelo", ln=True, align="C")

    # Adicionar tabela de métricas
    pdf.set_font("Arial", size=10)
    for i in range(len(metrics_df)):
        row = metrics_df.iloc[i]
        pdf.cell(200, 10, txt=str(row), ln=True)

    # Adicionar gráficos
    pdf.add_page()
    accuracy_fig.savefig("accuracy_plot.png", bbox_inches='tight')
    pdf.image("accuracy_plot.png", x=10, y=10, w=190)

    pdf.add_page()
    loss_fig.savefig("loss_plot.png", bbox_inches='tight')
    pdf.image("loss_plot.png", x=10, y=10, w=190)

    # Salvar o PDF em um arquivo temporário
    temp_pdf_file = tempfile.NamedTemporaryFile(delete=False)
    temp_pdf_file.close()  # Fecha o arquivo para que o FPDF possa acessá-lo

    pdf.output(temp_pdf_file.name)

    # Ler o conteúdo do arquivo temporário como bytes
    with open(temp_pdf_file.name, "rb") as f:
        pdf_bytes = f.read()

    # Remover o arquivo temporário
    os.remove(temp_pdf_file.name)

    return pdf_bytes



# Definindo os parâmetros e categorias do modelo
CATEGORIES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
IMG_SIZE = 100

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

# Etapa 1: Upload do dataset
uploaded_file = st.file_uploader("Faça o upload de um arquivo zip com o dataset", type="zip")

if uploaded_file is not None:
    # Salva o arquivo zip carregado
    with open("uploaded_dataset.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Upload concluído!")

    # Extrai o arquivo zip para um diretório
    with zipfile.ZipFile("uploaded_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("uploaded_dataset")
    st.success("Dataset extraído com sucesso!")

    path_test = "uploaded_dataset"

# Input para escolher o número de épocas
nb_epochs = st.number_input('Escolha o número de épocas', min_value=1, max_value=20, value=5)

# Botão para iniciar a análise e o treinamento do modelo
if st.button('Iniciar Análise e Treinamento'):
    timer_start = time.time()
    timer_running = True
    if path_test:
        # Etapa 2: Preparar conjunto de dados para treinamento
        st.write("Preparando conjunto de dados")
        st.write("STATUS: Processando...")
        path_test = os.path.join(path_test, "dataset2-master", "dataset2-master", "images", "TRAIN")
        transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
        dataset = load_image_data(path_test, transform)

        if not dataset:
            st.write("STATUS: ERRO")
            st.error("Erro ao carregar os dados. Verifique a estrutura do dataset.")
        else:
            # Dividir o dataset em treino e teste
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

            # Inicializar o modelo
            model = SimpleCNN(num_classes=4)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
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

                st.write(
                    f"Epoch {epoch + 1}/{nb_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
                st.markdown("STATUS: Concluido!")

            # Plotar a precisão e a perda
            st.write("Visualização da Acurácia e Perda")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(train_acc_history, label='Train Accuracy')
            ax[0].plot(val_acc_history, label='Validation Accuracy')
            ax[0].legend()
            ax[0].set_title('Accuracy')
            ax[1].plot(train_loss_history, label='Train Loss')
            ax[1].plot(val_loss_history, label='Validation Loss')
            ax[1].legend()
            ax[1].set_title('Loss')
            st.pyplot(fig)

            # Gerar dataframe com as métricas
            metrics_dict = {
                "Metric": ["Train Loss", "Train Accuracy", "Val Loss", "Final Val Accuracy"],
                "Value": [train_loss_history[-1], train_acc_history[-1], val_loss_history[-1], val_acc_history[-1]]
            }
            metrics_df = pd.DataFrame(metrics_dict)

            # Plotar as métricas
            st.write(metrics_df)

            # Gerar relatório em PDF
            pdf_report = generate_pdf_report(metrics_df, ax[0].figure, ax[1].figure)
            st.download_button(
                label="Download PDF Report",
                data=pdf_report,
                file_name="report.pdf",
                mime="application/pdf"
            )

            # Timer
            timer_running = False
            elapsed_time = (time.time() - timer_start)/60
            st.write(f"Tempo total de análise: {elapsed_time:.2f} minutos")
            st.success("Análise concluída!")

# Botão para limpar todos os dados da análise feita para realizar uma nova análise
if st.button('Limpar Dados'):
    # Remover arquivos e resetar variáveis
    if os.path.exists("uploaded_dataset.zip"):
        os.remove("uploaded_dataset.zip")
    if os.path.exists("uploaded_dataset"):
        for root, dirs, files in os.walk("uploaded_dataset", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir("uploaded_dataset")
    st.write("Todos os dados foram limpos.")
