# modules/DataLoader/dataloader.py

import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class RealWasteDataset(Dataset):
    """
    Dataset customizado para o dataset RealWaste.
    Esta classe assume a seguinte estrutura de diretórios:
    /caminho/para/o/dataset/
    ├── classe_1/
    │   ├── imagem1.jpg
    │   └── imagem2.jpg
    ├── classe_2/
    │   ├── imagem3.jpg
    │   └── imagem4.jpg
    ...
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Diretório principal do dataset contendo as subpastas das classes.
            transform (callable, optional): Transformações do torchvision a serem aplicadas na imagem.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        
        # Encontra as classes e cria um mapeamento de nome_da_classe para índice_numérico
        self.classes = sorted([d.name for d in os.scandir(data_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Carrega os caminhos das imagens e seus respectivos rótulos
        self.samples = self._load_samples()
        
        print(f"Dataset carregado. Encontradas {len(self.samples)} imagens em {len(self.classes)} classes.")
        print(f"Mapeamento de classes: {self.class_to_idx}")

    def _load_samples(self):
        """
        Método privado para percorrer as pastas e coletar os caminhos das imagens e seus rótulos.
        Retorna uma lista de tuplas (caminho_da_imagem, rótulo_numérico).
        """
        samples = []
        # Itera sobre cada pasta de classe (ex: 'paper', 'plastic')
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.data_dir, class_name)
            
            # Itera sobre cada arquivo na pasta da classe
            for file_name in os.listdir(class_dir):
                # Garante que está pegando apenas arquivos de imagem
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, file_name)
                    samples.append((image_path, class_idx))
                    
        return samples

    def __len__(self):
        """
        Retorna o número total de amostras no dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Busca e retorna uma amostra do dataset no índice `idx`.
        
        Args:
            idx (int): O índice da amostra a ser retornada.
            
        Returns:
            tuple: (imagem, rótulo)
        """
        image_path, label = self.samples[idx]
        
        try:
            # Carrega a imagem e a converte para RGB (garante 3 canais)
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {image_path}")
            return None, None

        # Aplica as transformações na imagem, se houver
        if self.transform:
            image = self.transform(image)
        
        # Converte o rótulo para um tensor PyTorch
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
