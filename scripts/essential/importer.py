import os
import sys
import shutil
import argparse
import importlib.util

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

def carregar_adaptador(caminho_script):
    spec = importlib.util.spec_from_file_location("importer_adapter", caminho_script)
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    return importer

def mover_ou_copiar_videos(pasta_origem, pasta_destino, nome_dataset, mode, adapter_name):
    adapter_location = "../../adapters/" + adapter_name
    extensoes_video = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
    videos_importados = []
    video_id = 1  # Inicia o contador de IDs

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    for raiz, _, arquivos in os.walk(pasta_origem):
        for arquivo in arquivos:
            nome, extensao = os.path.splitext(arquivo)
            extensao = extensao.lower()

            if extensao in extensoes_video:
                caminho_origem = os.path.join(raiz, arquivo)
                subpastas = os.path.relpath(raiz, pasta_origem).split(os.sep)

                # Novo nome baseado no dataset e no ID
                novo_nome = f"{nome_dataset}_{video_id}{extensao}"

                caminho_destino = os.path.join(pasta_destino, novo_nome)

                if not os.path.exists(caminho_destino):
                    if mode == "move":
                        shutil.move(caminho_origem, caminho_destino)
                        print(f"Movido: {arquivo} -> {novo_nome}")
                    elif mode == "copy":
                        shutil.copy2(caminho_origem, caminho_destino)
                        print(f"Copiado: {arquivo} -> {novo_nome}")
                    
                    videos_importados.append({
                        "nome_arquivo": nome,
                        "novo_nome": novo_nome,
                        "subpastas": subpastas,
                        "origem": caminho_origem,
                        "destino": caminho_destino
                    })

                    video_id += 1  # Incrementa o ID para o próximo vídeo
                else:
                    print(f"Arquivo já existe (ignorado): {novo_nome}")

    if adapter_location and os.path.isfile(adapter_location):
        adaptador = carregar_adaptador(adapter_location)
        adaptador.gerar_metadados(videos_importados, pasta_destino, nome_dataset)
    else:
        print("Adaptador não encontrado ou não especificado.")

    print("Concluído!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move ou copia vídeos e gera metadados com um adaptador externo"
    )
    parser.add_argument("origem", help="Diretório de origem")
    parser.add_argument("dataset", help="Nome do dataset (prefixo)")
    parser.add_argument("--mode", choices=["move", "copy"], default="copy",
                        help="Mover ou copiar arquivos (padrão: copy)")
    parser.add_argument("--adapter", default="generic_adapter.py",
                        help="Nome do script do adaptador (padrão: generic_adapter.py)")

    args = parser.parse_args()

    if not os.path.isdir(args.origem):
        print(f"Erro: '{args.origem}' não é um diretório válido.")
        sys.exit(1)

    pasta_destino = "../../data/datasets/video"
    mover_ou_copiar_videos(args.origem, pasta_destino, args.dataset, args.mode, args.adapter)
