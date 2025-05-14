'''adaptador genérico que utiliza somente a pasta onde os vídeos estão e os nomes dos vídeos para gerar as tags dos metadados'''

import os
import yaml

def gerar_metadados(videos_importados, destino, dataset):
    caminho_saida = os.path.join(destino, "../meta_video.yml")
    
    if os.path.exists(caminho_saida):
        with open(caminho_saida, "r", encoding="utf-8") as f:
            metadados = yaml.safe_load(f) or []
    else:
        metadados = []

    for video in videos_importados:
        tags = {"-name": video["nome_arquivo"]}
        
        for i, pasta in enumerate(video["subpastas"], start=1):
            tags[f"-folder{i}"] = pasta
        
        metadados.append({
            "video": video["novo_nome"],  # Usando o novo_nome em vez do índice
            "tags": tags,
        })

    with open(caminho_saida, "w", encoding="utf-8") as f:
        yaml.dump(metadados, f, allow_unicode=True, sort_keys=False)

    print(f"Arquivo de metadados {'criado' if not os.path.exists(caminho_saida) else 'atualizado'}: {caminho_saida}")