# experiments/ResNet/demo_dataloader.py

import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.DataLoader.dataloader import RealWasteDataset

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def visualize_batch(images, labels, class_names):
    """Visualiza um lote de imagens com seus rótulos."""
    plt.figure(figsize=(15, 10))
    plt.suptitle("Demonstração do DataLoader - Lote de Amostras", fontsize=16)
    
   
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(min(len(images), 8)): 
        ax = plt.subplot(2, 4, i + 1)
        
        # Inverte a transformação para visualização
        img = images[i].permute(1, 2, 0).numpy() # Converte de (C, H, W) para (H, W, C)
        img = std * img + mean # Desnormaliza
        img = np.clip(img, 0, 1) # Garante que os valores dos pixels estão entre 0 e 1
        
        plt.imshow(img)
        
        # Usa a lista de classes do dataset para obter o nome do rótulo
        label_name = class_names[labels[i].item()]
        plt.title(f"Label: {label_name}")
        plt.axis("off")
    
    # Salva a figura em vez de tentar mostrá-la
    save_path = "experiments/batch_visualization.png"
    plt.savefig(save_path)
    print(f"\nVisualização do lote salva em '{save_path}'")

def main():
    print("Iniciando a demonstração do DataLoader para o dataset RealWaste...")
    
    # Carrega a configuração
    config = load_config()
    dataset_config = config['dataset']
    model_config = config['model']
    
    # Define o pipeline de transformações de imagem
    image_transforms = transforms.Compose([
        transforms.Resize((model_config['image_size'], model_config['image_size'])),
        transforms.ToTensor(), # Converte para Tensor e normaliza pixels para [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalização padrão da ImageNet
    ])
    
    # Instancia o Dataset customizado
    print(f"Carregando dataset do caminho: {dataset_config['path']}")
    full_dataset = RealWasteDataset(data_dir=dataset_config['path'], transform=image_transforms)
    
    # Instancia o DataLoader do PyTorch
    data_loader = DataLoader(
        dataset=full_dataset,
        batch_size=dataset_config['batch_size'],
        shuffle=True # Embaralha os dados
    )
    
    print("\n--- Informações do DataLoader ---")
    print(f"Tamanho total do dataset: {len(full_dataset)}")
    print(f"Classes encontradas: {full_dataset.classes}")
    print(f"Tamanho do lote (batch size): {dataset_config['batch_size']}")
    print(f"Número de lotes por época: {len(data_loader)}")
    
    # Pega um lote de dados para verificar
    print("\nBuscando um lote de dados para teste...")
    images, labels = next(iter(data_loader))
    
    print("\n--- Verificação do Lote ---")
    print(f"Formato (shape) do lote de imagens: {images.shape}") # Deve ser [batch_size, canais, altura, largura]
    print(f"Formato (shape) do lote de rótulos: {labels.shape}") # Deve ser [batch_size]
    print(f"Exemplo de rótulos no lote: {labels.tolist()}")
    
    # Visualiza o lote
    visualize_batch(images, labels, full_dataset.classes)

if __name__ == '__main__':
    main()