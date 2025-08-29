# modules/Evaluation/evaluator.py

import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np
import pandas as pd
import os
import json 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from minisom import MiniSom

from modules.Preprocessing.transforms import get_image_transforms
from modules.Utils.utils import EarlyStopping
from modules.DataLoader.dataloader import RealWasteDataset
from models.resnetbaseline import ResNetBaseline

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calculate_macro_specificity(y_true, y_pred, num_classes):
    """
    NOVA FUNÇÃO: Calcula a especificidade para cada classe e retorna a média (macro).
    """
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    class_specificities = []
    
    for i in range(num_classes):
        # Verdadeiros Negativos (TN) para a classe i
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        
        # Falsos Positivos (FP) para a classe i
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        # Calcula a especificidade
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        class_specificities.append(specificity)
        
    # Retorna a média macro (média simples entre as classes)
    return np.mean(class_specificities)

def plot_and_save_learning_curves(train_history, val_history, fold_num, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Perda de Treino')
    plt.plot(val_history, label='Perda de Validação')
    plt.title(f'Curvas de Aprendizagem - Fold {fold_num}')
    plt.xlabel('Épocas'); plt.ylabel('Perda (Loss)'); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, "learning_curves.png")); plt.close()

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, title_suffix, fold_num, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Matriz de Confusão - Fold {fold_num} ({title_suffix})', fontsize=16)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axs[0])
    axs[0].set_title('Contagens Absolutas'); axs[0].set_ylabel('Classe Verdadeira'); axs[0].set_xlabel('Classe Prevista')
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='viridis', xticklabels=class_names, yticklabels=class_names, ax=axs[1])
    axs[1].set_title('Percentual por Classe (Sensibilidade)'); axs[1].set_ylabel('Classe Verdadeira'); axs[1].set_xlabel('Classe Prevista')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{title_suffix.lower()}.png")); plt.close()

def plot_som_for_fold(som, features, labels, class_names, fold_num, save_dir):
    fig, axs = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw={'width_ratios': [4, 1]})
    fig.suptitle(f'SOM do Fold de Validação {fold_num}', fontsize=16)
    ax_map = axs[0]
    im = ax_map.pcolor(som.distance_map().T, cmap='bone_r')
    num_classes = len(class_names)
    colors = plt.cm.get_cmap('tab10', num_classes)
    all_markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>']
    markers = all_markers[:num_classes]
    for i, class_name in enumerate(class_names):
        class_features = features[labels == i]
        if len(class_features) > 0:
            winner_coords = np.array([som.winner(feat) for feat in class_features])
            jitter = np.random.uniform(-0.25, 0.25, size=winner_coords.shape)
            ax_map.plot(winner_coords[:, 0] + 0.5 + jitter[:, 0], winner_coords[:, 1] + 0.5 + jitter[:, 1], markers[i], markerfacecolor=colors(i), markeredgecolor='k', markersize=10, markeredgewidth=0.5, linestyle='None')
    ax_map.set_xlabel('Coordenada X do Neurônio'); ax_map.set_ylabel('Coordenada Y do Neurônio')
    ax_map.grid(True, linestyle='--', alpha=0.5); fig.colorbar(im, ax=ax_map, label='Distância Inter-neuronal (U-Matrix)')
    ax_legend = axs[1]
    legend_elements = [plt.Line2D([0], [0], marker=markers[i], color='w', label=class_names[i], markerfacecolor=colors(i), markeredgecolor='k', markersize=12, linestyle='None') for i in range(num_classes)]
    ax_legend.legend(handles=legend_elements, loc='center', title="Classes", fontsize=12, title_fontsize=14)
    ax_legend.axis('off')
    fig.savefig(os.path.join(save_dir, "som_visualization_final.png")); plt.close(fig)

def evaluate_unsupervised_with_supervised_metrics(som_clusters, true_labels, class_names, fold_num, save_dir):
    df = pd.DataFrame({'true_label': true_labels, 'cluster': som_clusters})
    contingency_table = pd.crosstab(df['true_label'], df['cluster'])
    cluster_to_label_map = contingency_table.idxmax(axis=0).to_dict()
    predicted_labels = df['cluster'].map(cluster_to_label_map).astype(int)
    accuracy = accuracy_score(df['true_label'], predicted_labels)
    report_dict = classification_report(df['true_label'], predicted_labels, labels=np.arange(len(class_names)), target_names=class_names, zero_division=0, output_dict=True)
    report_path = os.path.join(save_dir, "classification_report_unsupervised.json")
    with open(report_path, 'w') as f: json.dump(report_dict, f, indent=4)
    plot_and_save_confusion_matrix(df['true_label'], predicted_labels, class_names, "SOM_Mapeado", fold_num, save_dir)
    specificity = calculate_macro_specificity(df['true_label'], predicted_labels, len(class_names))
    return report_dict, specificity

def extract_features_from_model(model, data_loader, device):
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for data, labels in tqdm(data_loader, desc="Extraindo Features"):
            features = model.extract_features(data.to(device))
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)

# =============================================================================
# FUNÇÃO PRINCIPAL DO EVALUATOR 
# =============================================================================

def run_kfold_analysis(config, experiment_dir):
    # ... (Seção de setup permanece a mesma) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_dataset = RealWasteDataset(data_dir=config['dataset']['path'])
    num_classes, class_names = len(full_dataset.classes), full_dataset.classes
    print(f"Dataset carregado com {num_classes} classes.")
    k_folds = config['cross_validation']['n_splits']
    y_labels = [sample[1] for sample in full_dataset.samples]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    sup_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'specificity': []}
    unsup_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'specificity': []}

    for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):
        # ... (Loop de treinamento supervisionado) ...
        fold_num = fold + 1
        print(f"\n{'='*20} Fold {fold_num}/{k_folds} {'='*20}")
        fold_dir = os.path.join(experiment_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        train_subset, val_subset = Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)
        train_subset.dataset.transform = get_image_transforms(config['model']['image_size'], True)
        val_subset.dataset.transform = get_image_transforms(config['model']['image_size'], False)
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
        model = ResNetBaseline(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=config['training']['learning_rate'], weight_decay=float(config['training']['weight_decay']))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config['training']['scheduler_patience'])
        early_stopper = EarlyStopping(patience=config['training']['early_stopping_patience'], verbose=True, path=os.path.join(fold_dir, 'best_model.pth'))
        train_loss_history, val_loss_history = [], []
        for epoch in range(config['training']['epochs']):
            model.train(); train_loss = 0.0
            pbar_train = tqdm(train_loader, desc=f"Fold {fold_num} Epoch {epoch+1} [Treino]")
            for data, labels in pbar_train:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad(); outputs = model(data); loss = criterion(outputs, labels)
                loss.backward(); optimizer.step(); train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader); train_loss_history.append(avg_train_loss)
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device); outputs = model(data)
                    val_loss += criterion(outputs, labels).item()
            avg_val_loss = val_loss / len(val_loader); val_loss_history.append(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']; pbar_train.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss, lr=current_lr)
            scheduler.step(avg_val_loss); early_stopper(avg_val_loss, model)
            if early_stopper.early_stop: print(f"Early stopping ativado na época {epoch+1}"); break
        plot_and_save_learning_curves(train_loss_history, val_loss_history, fold_num, fold_dir)
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pth')))
        y_true, y_pred = [], []
        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data.to(device)); _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy()); y_pred.extend(predicted.cpu().numpy())
        report_sup = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)
        sup_metrics['accuracy'].append(report_sup['accuracy']); sup_metrics['precision'].append(report_sup['macro avg']['precision'])
        sup_metrics['recall'].append(report_sup['macro avg']['recall']); sup_metrics['f1-score'].append(report_sup['macro avg']['f1-score'])
        sup_metrics['specificity'].append(calculate_macro_specificity(y_true, y_pred, num_classes))
        with open(os.path.join(fold_dir, "classification_report_supervised.json"), 'w') as f: json.dump(report_sup, f, indent=4)
        plot_and_save_confusion_matrix(y_true, y_pred, class_names, "Supervisionado", fold_num, fold_dir)

        # --- Etapa 2: Análise Não Supervisionada ---
        print(f"\n--- Análise Não Supervisionada (SOM) para o Fold {fold_num} ---")
        features, labels = extract_features_from_model(model, val_loader, device)
        
        #Lendo parâmetros do SOM a partir do config ---
        som_config = config['som']
        map_size = (som_config['map_size_x'], som_config['map_size_y'])
        
        som = MiniSom(map_size[0], map_size[1], 
                      features.shape[1], 
                      sigma=som_config['sigma'], 
                      learning_rate=som_config['learning_rate'],
                      random_seed=config['dataset']['random_seed'])
        
        print("Treinando o SOM...")
        som.train_random(features, som_config['train_iterations'], verbose=True)
        # -----------------------------------------------------------------

        plot_som_for_fold(som, features, labels, class_names, fold_num, fold_dir)
        winner_coords = np.array([som.winner(x) for x in features]).T
        cluster_index = np.ravel_multi_index(winner_coords, map_size)
        report_unsup, spec_unsup = evaluate_unsupervised_with_supervised_metrics(cluster_index, labels, class_names, fold_num, fold_dir)
        
        unsup_metrics['accuracy'].append(report_unsup['accuracy']); unsup_metrics['precision'].append(report_unsup['macro avg']['precision'])
        unsup_metrics['recall'].append(report_unsup['macro avg']['recall']); unsup_metrics['f1-score'].append(report_unsup['macro avg']['f1-score'])
        unsup_metrics['specificity'].append(spec_unsup)

    
    final_results = {}
    for model_name, metrics_dict in [("resnet_supervised", sup_metrics), ("som_unsupervised", unsup_metrics)]:
        final_results[model_name] = {};
        for metric_name, values in metrics_dict.items():
            final_results[model_name][f"mean_{metric_name}"] = float(np.mean(values)); final_results[model_name][f"std_{metric_name}"] = float(np.std(values))
    print("\n" + "="*50); print("RESULTADO FINAL DA ANÁLISE COMPARATIVA"); print("-" * 50)
    for model_name, results in final_results.items():
        print(f"Modelo: {model_name}");
        for metric_name, value in results.items(): print(f"  - {metric_name}: {value:.4f}")
    print("=" * 50)
    
    return final_results