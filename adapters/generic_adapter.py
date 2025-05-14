import os
import yaml

def gerar_metadados(videos_importados, destino, dataset):
    metadados = []

    for video in videos_importados:
        metadados.append({
            "arquivo": video["nome_arquivo"],
            "tags": video["subpastas"],
            "dataset": dataset
        })

    caminho_saida = os.path.join(destino, f"{dataset}_metadados.yml")
    
    with open(caminho_saida, "w", encoding="utf-8") as f:
        yaml.dump(metadados, f, allow_unicode=True, sort_keys=False)

    print(f"Arquivo de metadados criado: {caminho_saida}")
