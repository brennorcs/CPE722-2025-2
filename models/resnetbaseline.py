# modules/Models/resnetbaseline.py
import torch
import torch.nn as nn
from torchvision import models


class ResNetBaseline(nn.Module):
    """
    Classe que encapsula um modelo ResNet18 pré-treinado para transfer learning.
    A classe define a ARQUITETURA. A responsabilidade de movê-la para o 
    dispositivo correto (CPU/GPU) é do script de treinamento.
    """
    def __init__(self, num_classes, pretrained=True):
        """
        Args:
            num_classes (int): O número de classes de saída. Este valor é
                               determinado dinamicamente a partir do dataset
                               pelo script de avaliação.
            pretrained (bool): Se True, carrega os pesos pré-treinados no ImageNet.
        """
        super(ResNetBaseline, self).__init__()
        
        # 1. Carrega o modelo ResNet-18 com os pesos especificados
        print("Inicializando ResNet-18...")
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        # 2. "Congela" os pesos das camadas convolucionais
        print("Congelando camadas de extração de features...")
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # 3. Substitui a camada final (o "classificador")
        # Apenas os parâmetros desta nova camada terão requires_grad = True por padrão
        print(f"Substituindo a camada de classificação para {num_classes} classes.")
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        """Define a passagem para frente (forward pass) para o treinamento."""
        return self.resnet(x)

    def extract_features(self, x):
        """
        Extrai os vetores de características da penúltima camada (antes do classificador).
        Essencial para a análise não supervisionada (SOM).
        """
        # Remove temporariamente a camada de classificação para pegar as features
        original_fc = self.resnet.fc
        self.resnet.fc = nn.Identity()
        
        # Passa os dados pelo modelo para extrair as features
        features = self.resnet(x)
        
        # Restaura a camada de classificação original
        self.resnet.fc = original_fc
        
        return features


