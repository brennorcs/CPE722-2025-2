# experiments/ResNet/run_experiment.py

import yaml
import sys
import os
from datetime import datetime
import shutil
import json

# Adiciona o diretório raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.Evaluation.evaluator import run_kfold_analysis

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()

    # Cria uma pasta única para este experimento com base no tempo
    model_name = "ResNet18_SOM_Analysis" # Nome mais descritivo para o experimento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/{model_name}/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Artefatos deste experimento serão salvos em: {experiment_dir}")

    # Salva uma cópia da configuração usada para reprodutibilidade
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    shutil.copy(os.path.join(project_root, 'config.yaml'), os.path.join(experiment_dir, 'config.yaml'))
    
    # 3. Chama a nova função com os argumentos corretos
    results = run_kfold_analysis(
        config=config,
        experiment_dir=experiment_dir 
    )

    results_path = os.path.join(experiment_dir, 'summary_results.json')
    
    # 2. Usa json.dump para salvar o arquivo
    #    O argumento 'indent=4' formata o arquivo para ser legível por humanos
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResultados de sumarização salvos em: {results_path}")


if __name__ == '__main__':
    main()