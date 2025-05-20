import argparse
import csv
import shutil
from pathlib import Path
import subprocess


VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.mpeg', '.mpg'}
METADATA_LOCATION = Path("../metadata/1_videos.csv")
DATA_LOCATION = Path("../data/1_videos")


def is_video_file(file_path):
    return file_path.suffix.lower() in VIDEO_EXTENSIONS


def get_video_duration(filepath):
    """
    Usa ffprobe para obter a duração do vídeo em segundos (float).
    Retorna string formatada em segundos com 2 casas decimais, ou '' se erro.
    """
    try:
        # Chama ffprobe para obter a duração em segundos
        result = subprocess.run(
            [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(filepath)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        duration_str = result.stdout.strip()
        duration = float(duration_str)
        return f"{duration:.2f}"
    except Exception:
        return ''

def get_all_existing_data(csv_path):
    """
    Lê o CSV e retorna:
    - linhas originais como listas
    - conjunto de datasets
    - dict {dataset: max_level}
    """
    if not csv_path.exists():
        return [], set(), {}

    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = list(csv.reader(f))
        if not reader:
            return [], set(), {}

        header = reader[0]
        data_rows = reader[1:]

        datasets = set()
        dataset_levels = {}

        # A partir do header, identificar colunas de levels e a quais datasets pertencem
        # As 3 primeiras colunas fixas: id, dataset, name
        # Agora a 4ª fixa: duration
        # Depois colunas do tipo <dataset>_level_X

        for col_name in header[4:]:
            # ex: dataset1_level_1
            if '_level_' in col_name:
                ds, _, level_str = col_name.rpartition('_level_')
                level = int(level_str)
                datasets.add(ds)
                if ds not in dataset_levels or dataset_levels[ds] < level:
                    dataset_levels[ds] = level

        return data_rows, datasets, dataset_levels


def build_header(datasets, dataset_levels, current_dataset, current_max_level):
    """
    Constrói o header final:
    id, dataset, name, duration, colunas de todos datasets (com os níveis máximos conhecidos),
    adicionando níveis do dataset atual se forem maiores que os já conhecidos.
    """

    # Atualiza o max level do dataset atual, se maior
    if current_dataset in dataset_levels:
        if dataset_levels[current_dataset] < current_max_level:
            dataset_levels[current_dataset] = current_max_level
    else:
        dataset_levels[current_dataset] = current_max_level
        datasets.add(current_dataset)

    header = ['id', 'dataset', 'name', 'duration']
    # Ordena os datasets alfabeticamente para manter consistência (pode trocar se quiser)
    for ds in sorted(datasets):
        max_lvl = dataset_levels.get(ds, 0)
        for lvl in range(1, max_lvl + 1):
            header.append(f"{ds}_level_{lvl}")

    return header, dataset_levels


def main():
    parser = argparse.ArgumentParser(
        description='Importar vídeos de um diretório (e subdiretórios) e atualizar o arquivo videos.csv.',
        epilog='''\
Exemplo de uso:
  python generic_importer.py meu_dataset /caminho/do/dataset --folder --move

Argumentos obrigatórios:
  meu_dataset             Nome do dataset a ser importado
  /caminho/do/dataset     Caminho para o diretório onde estão os vídeos

Opções:
  --folder                Salvar nome das subpastas (todos os níveis) como colunas <dataset>_level_X
  --move                  Move os vídeos para a pasta importada renomeando para o id
  --copy                  Copia os vídeos para a pasta importada renomeando para o id
''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('dataset', help='Nome do dataset')
    parser.add_argument('directory', help='Caminho do diretório do dataset')
    parser.add_argument('--folder', action='store_true', help='Incluir todos os níveis de subpasta como colunas')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--move', action='store_true', help='Move os vídeos em vez de apenas listá-los')
    group.add_argument('--copy', action='store_true', help='Copia os vídeos em vez de apenas listá-los')

    args = parser.parse_args()
    dataset_name = args.dataset
    base_path = Path(args.directory).resolve()
    csv_path = METADATA_LOCATION.resolve()
    import_dest = DATA_LOCATION.resolve()
    import_dest.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Primeiro, coletar todos os vídeos para descobrir a profundidade máxima dos diretórios
    video_files = []
    max_depth = 0

    for filepath in base_path.rglob('*'):
        if filepath.is_file() and is_video_file(filepath):
            video_files.append(filepath)
            try:
                relative_path_parts = filepath.relative_to(base_path).parts[:-1]  # pastas até o vídeo
                depth = len(relative_path_parts)
                if depth > max_depth:
                    max_depth = depth
            except ValueError:
                pass

    # Ler dados antigos e informações sobre datasets e níveis
    old_rows, existing_datasets, dataset_levels = get_all_existing_data(csv_path)

    # Construir header global com todos datasets e níveis
    header, dataset_levels = build_header(existing_datasets, dataset_levels, dataset_name, max_depth)

    # Mapear índices das colunas antigas para adaptar linhas antigas
    old_header = []
    if csv_path.exists():
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            try:
                old_header = next(reader)
            except StopIteration:
                old_header = []

    old_header_index = {col: idx for idx, col in enumerate(old_header)}

    # Adaptar linhas antigas para o novo header
    adapted_old_rows = []
    for row in old_rows:
        new_row = [''] * len(header)
        # Colunas fixas:
        for col in ['id', 'dataset', 'name', 'duration']:
            if col in old_header_index and old_header_index[col] < len(row):
                new_row[header.index(col)] = row[old_header_index[col]]
            else:
                # Para 'duration' se não existir, colocar '-'
                if col == 'duration':
                    new_row[header.index(col)] = '-'
                else:
                    new_row[header.index(col)] = ''

        # Para as colunas de levels, preencher com dado antigo ou '-'
        for col in header[4:]:
            if col in old_header_index and old_header_index[col] < len(row):
                new_row[header.index(col)] = row[old_header_index[col]]
            else:
                new_row[header.index(col)] = '-'

        adapted_old_rows.append(new_row)

    # Próximo id: máximo dos ids antigos +1, ou 1 se vazio
    if adapted_old_rows:
        last_id = max(int(r[0]) for r in adapted_old_rows if r[0].isdigit())
        next_id = last_id + 1
    else:
        next_id = 1

    new_rows = []

    for filepath in video_files:
        new_row = [''] * len(header)
        new_row[header.index('id')] = str(next_id)
        new_row[header.index('dataset')] = dataset_name
        new_row[header.index('name')] = filepath.name

        # Obter duração do vídeo
        duration = get_video_duration(filepath)
        new_row[header.index('duration')] = duration if duration else '-'

        try:
            relative_path_parts = filepath.relative_to(base_path).parts[:-1]
        except ValueError:
            relative_path_parts = []

        # Preencher níveis
        for col in header[4:]:
            ds, _, lvl_str = col.rpartition('_level_')
            lvl = int(lvl_str)
            if ds == dataset_name:
                if lvl <= len(relative_path_parts):
                    new_row[header.index(col)] = relative_path_parts[lvl - 1]
                else:
                    new_row[header.index(col)] = ''
            else:
                new_row[header.index(col)] = '-'

        new_rows.append(new_row)

        # Renomear com ID e mover/copiar
        ext = filepath.suffix.lower()
        new_filename = f"{next_id}{ext}"
        destination = import_dest / new_filename

        if args.move:
            shutil.move(str(filepath), str(destination))
        elif args.copy:
            shutil.copy2(str(filepath), str(destination))

        next_id += 1

    # Escrever tudo no CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in adapted_old_rows:
            writer.writerow(r)
        for r in new_rows:
            writer.writerow(r)

    print(f"[✅] {len(new_rows)} vídeos importados para {csv_path}")


if __name__ == '__main__':
    main()

