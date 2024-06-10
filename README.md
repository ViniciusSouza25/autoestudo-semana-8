# Pose Estimation para Bovinos

## Introdução

Este projeto visa implementar um sistema de pose estimation para bovinos utilizando o dataset ANIMAL-POSE. o projeto se concentra na extração, visualização, filtragem manual e processamento das imagens.

## 1. Obtenção do Dataset e Análise Exploratória

### Download e Extração do Dataset

O dataset foi baixado e extraído do Google Drive utilizando os seguintes comandos:

```python
from google.colab import drive
import zipfile
import os

# Montar o Google Drive
drive.mount('/content/drive')

# Diretório onde o dataset será salvo após extração
output_dir = "/content/animal_pose_dataset"
zip_path = "/content/drive/MyDrive/animal_pose_dataset/Copy of images.zip"

# Verificar se o arquivo ZIP existe
if not os.path.exists(zip_path):
    raise Exception(f"Arquivo ZIP não encontrado no caminho: {zip_path}")

# Extrair o conteúdo
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print("Extração concluída.")

```

### Listagem e Visualização das Imagens

Após a extração, listamos e visualizamos algumas imagens para entender o conteúdo do dataset.

```python
import matplotlib.pyplot as plt
import cv2

# Função para mostrar uma imagem
def show_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Listar arquivos de imagem
image_files = [os.path.join(root, file) for root, _, files in os.walk(output_dir) for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]

# Mostrar as primeiras 5 imagens
for image_path in image_files[:5]:
    show_image(image_path)
```


## Filtragem do Dataset para Imagens de Bovinos

Como não há anotações disponíveis, a filtragem para imagens de bovinos será feita manualmente.

```python
# Suponha que você identificou manualmente as imagens de bovinos
bovino_image_files = [
    # Lista de caminhos das imagens de bovinos identificadas
]

print(f"Total de imagens de bovinos identificadas: {len(bovino_image_files)}")



```
## Processamento de Imagens

Vamos processar as imagens de bovinos para prepará-las para o modelo de pose estimation.

```python
# Diretório para salvar as imagens de bovinos processadas
bovino_processed_dir = os.path.join(output_dir, "bovino_processed")
os.makedirs(bovino_processed_dir, exist_ok=True)

# Função para redimensionar e normalizar as imagens
def process_image(image_path, output_size=(256, 256)):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, output_size)
    normalized_image = resized_image / 255.0
    return normalized_image

# Processar e salvar as imagens de bovinos
for image_path in bovino_image_files:
    processed_image = process_image(image_path)
    output_image_path = os.path.join(bovino_processed_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, (processed_image * 255).astype('uint8'))

print("Processamento de imagens de bovinos concluído.")

```


## Resultados Finais

Apresentamos os resultados finais do processamento das imagens de bovinos.

```python
import pandas as pd

# Tabela de resumo das imagens processadas
summary_data = {
    "Total de Imagens de Bovinos": [len(bovino_image_files)],
    "Tamanho das Imagens Processadas": ["256x256"]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df)

```
