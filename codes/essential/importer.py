import os
import sys
import shutil
import argparse

def mover_videos(pasta_origem, pasta_destino, nome_dataset):
    extensoes_video = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
    
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    for raiz, _, arquivos in os.walk(pasta_origem):
        for arquivo in arquivos:
            nome, extensao = os.path.splitext(arquivo)
            extensao = extensao.lower()

            if extensao in extensoes_video:
                caminho_origem = os.path.join(raiz, arquivo)
                subpastas = os.path.relpath(raiz, pasta_origem).split(os.sep)
                
                # Novo formato: <dataset>_<nome>_<subpastas>
                partes_nome = [nome_dataset, nome]
                if subpastas != ['.']:
                    partes_nome.extend(subpastas)
                novo_nome = "_".join(partes_nome) + extensao

                caminho_destino = os.path.join(pasta_destino, novo_nome)

                if not os.path.exists(caminho_destino):
                    shutil.move(caminho_origem, caminho_destino)
                    print(f"Movido: {arquivo} -> {novo_nome}")
                else:
                    print(f"Arquivo já existe (ignorado): {novo_nome}")

    print("Concluído!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move vídeos de uma pasta origem para destino, renomeando com formato: <dataset>_<nome>_<subpastas>"
    )
    parser.add_argument("origem", help="Diretório de origem para buscar vídeos")
    parser.add_argument("dataset", help="Nome do dataset (prefixo dos arquivos)")
    args = parser.parse_args()

    if not os.path.isdir(args.origem):
        print(f"Erro: '{args.origem}' não é um diretório válido.")
        sys.exit(1)

    # Define o destino fixo (modifique conforme necessário)
    pasta_destino = "../../data/datasets/video"
    mover_videos(args.origem, pasta_destino, args.dataset)