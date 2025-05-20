import os

# Estrutura de diretórios e arquivos
structure = {
    "data": ["1_videos", "2_keypoints", "3_models", "4_metrics"],
    "lists": ["1_videos", "2_keypoints", "3_models", "4_metrics"],
    "metadata": ["1_videos.csv", "2_keypoints.csv", "3_models.csv", "4_metrics.csv"]
}

# Cabeçalhos padrão por arquivo de metadados (se quiser adicionar mais no futuro)
headers = {
    "1_videos.csv": "id,dataset,name,duration\n"
}

# Criação dos diretórios e arquivos
for base_folder, items in structure.items():
    base_path = os.path.join(os.getcwd(), base_folder)
    os.makedirs(base_path, exist_ok=True)
    for item in items:
        path = os.path.join(base_path, item)
        if item.endswith(".csv"):
            # Só cria o arquivo se ele ainda não existir
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(headers.get(item, ""))  # adiciona cabeçalho se existir
        else:
            os.makedirs(path, exist_ok=True)

print("Estrutura criada com sucesso (sem sobrescrever arquivos existentes).")
