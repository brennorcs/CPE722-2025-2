# modules/Preprocessing/transforms.py

from torchvision import transforms

def get_image_transforms(image_size, is_train=True):
    """
    Retorna um pipeline de transformações do torchvision.
    
    Args:
        image_size (int): O tamanho (altura e largura) para redimensionar a imagem.
        is_train (bool): Se True, retorna transformações com data augmentation para treino.
                         Se False, retorna transformações para validação/teste.
                         
    Returns:
        torchvision.transforms.Compose: O pipeline de transformações.
    """
    # Média e desvio padrão do ImageNet, padrão para transfer learning
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        # Pipeline de treino com Data Augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Pipeline de validação/teste sem augmentation
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.1)), # Redimensiona um pouco maior
            transforms.CenterCrop(image_size),       # Corta o centro no tamanho exato
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])