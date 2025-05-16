import os
import sys
import shutil
import argparse
import csv
from pathlib import Path

EXTENSOES_VIDEO = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
PASTA_DESTINO = Path("../data/datasets/video")
PASTA_METADATA = Path("../data/datasets/metadata")
PASTA_LISTAS = Path("../data/lists")

def importar_videos(origem: Path, dataset: str, modo: str):
    PASTA_DESTINO.mkdir(parents=True, exist_ok=True)
    PASTA_METADATA.mkdir(parents=True, exist_ok=True)
    PASTA_LISTAS.mkdir(parents=True, exist_ok=True)

    nome_arquivo_metadados = f"meta_{dataset}.csv"
    nome_arquivo_listas = f"list_all_{dataset}.csv"
    caminho_csv_metadata = PASTA_METADATA / nome_arquivo_metadados
    caminho_csv_listas = PASTA_LISTAS / nome_arquivo_listas

    metadados = []
    video_id = len([f for f in PASTA_DESTINO.iterdir() if f.suffix in EXTENSOES_VIDEO and f.name.startswith(dataset)]) + 1

    for raiz, _, arquivos in os.walk(origem):
        for nome in arquivos:
            caminho = Path(raiz) / nome
            if caminho.suffix.lower() not in EXTENSOES_VIDEO:
                continue

            novo_nome = f"{dataset}_{video_id}{caminho.suffix.lower()}"
            destino = PASTA_DESTINO / novo_nome

            if destino.exists():
                print(f"Ignorado (já existe): {novo_nome}")
                continue

            (shutil.move if modo == "move" else shutil.copy2)(caminho, destino)
            print(f"{'Movido' if modo == 'move' else 'Copiado'}: {caminho} -> {novo_nome}")

            subpastas = caminho.relative_to(origem).parent.parts
            linha = {"video": novo_nome, "-name": caminho.stem}
            linha.update({f"-folder{i+1}": p for i, p in enumerate(subpastas)})

            metadados.append(linha)
            video_id += 1

    salvar_metadados_csv([caminho_csv_metadata, caminho_csv_listas], metadados)
    print("Concluído!")

def salvar_metadados_csv(caminhos_csv: list[Path], metadados):
    if not metadados:
        print("Nenhum metadado para salvar.")
        return

    colunas = sorted({k for item in metadados for k in item.keys()})
    for caminho_csv in caminhos_csv:
        with open(caminho_csv, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["video"] + [c for c in colunas if c != "video"])
            writer.writeheader()
            for item in metadados:
                writer.writerow(item)
        print(f"Metadados salvos em CSV: {caminho_csv}")

def main():
    parser = argparse.ArgumentParser(description="Move ou copia vídeos e gera metadados.")
    parser.add_argument("origem", help="Diretório de origem")
    parser.add_argument("dataset", help="Nome do dataset (prefixo)")
    parser.add_argument("--mode", choices=["move", "copy"], default="copy")
    args = parser.parse_args()

    origem = Path(args.origem)
    if not origem.is_dir():
        print(f"Erro: '{origem}' não é um diretório válido.")
        sys.exit(1)

    importar_videos(origem, args.dataset, args.mode)

if __name__ == "__main__":
    main()
