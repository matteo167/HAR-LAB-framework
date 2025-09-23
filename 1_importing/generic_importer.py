# generic importer for testing
# Example: python3 importer.py teste /home/matteo/matteo/datasets/Teste --folder --copy

import argparse
import shutil
import subprocess
from pathlib import Path
import pandas as pd

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.mpeg', '.mpg'}
METADATA_VIDEO = Path("../metadata/1_videos.csv")
DATA_VIDEO = Path("../data/1_videos")

def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS

def get_video_duration(filepath: Path) -> str:
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        return f"{float(result.stdout.strip()):.2f}"
    except Exception:
        return ''

def read_metadata(csv_path: Path) -> pd.DataFrame:
    """Lê o CSV de metadados ou cria um DataFrame vazio com as colunas padrão."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=['id_video', 'dataset', 'name', 'duration', 'extension'])

def generate_level_columns(max_level: int, prefix: str) -> list[str]:
    """Gera nomes de colunas para cada nível de subpasta."""
    return [f"{prefix}_level_{i}" for i in range(1, max_level + 1)]

def process_videos(base_path: Path) -> tuple[list[Path], list[list[str]], int]:
    """Retorna lista de vídeos, partes do caminho e profundidade máxima."""
    video_files = [p for p in base_path.rglob('*') if p.is_file() and is_video_file(p)]
    paths_parts = [list(p.relative_to(base_path).parts[:-1]) for p in video_files]
    max_depth = max((len(parts) for parts in paths_parts), default=0)
    return video_files, paths_parts, max_depth

def main():
    parser = argparse.ArgumentParser(description='Importa vídeos e atualiza o CSV de metadados.')
    parser.add_argument('dataset', help='Nome do dataset')
    parser.add_argument('directory', help='Caminho do diretório com os vídeos')
    parser.add_argument('--folder', action='store_true', help='Incluir níveis de subpastas como colunas')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--move', action='store_true', help='Move os vídeos')
    group.add_argument('--copy', action='store_true', help='Copia os vídeos')
    args = parser.parse_args()

    dataset = args.dataset
    base_path = Path(args.directory).resolve()
    metadata_path = METADATA_VIDEO.resolve()
    dest_path = DATA_VIDEO.resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Lê o CSV antigo e define o próximo id_video disponível
    df_old = read_metadata(metadata_path)
    current_id_video = df_old['id_video'].astype(int).max() + 1 if not df_old.empty else 1

    video_files, paths_parts, max_depth = process_videos(base_path)
    level_cols = generate_level_columns(max_depth, dataset) if args.folder else []
    base_cols = ['id_video', 'dataset', 'name', 'duration', 'extension']

    records: list[dict] = []
    for i, filepath in enumerate(video_files):
        parts = paths_parts[i]
        ext = filepath.suffix.lower()
        new_name = f"{current_id_video}{ext}"
        dest_file = dest_path / new_name

        if args.move:
            shutil.move(str(filepath), str(dest_file))
        else:
            shutil.copy2(str(filepath), str(dest_file))

        record = {
            'id_video': current_id_video,
            'dataset': dataset,
            'name': filepath.name,
            'duration': get_video_duration(filepath),
            'extension': ext
        }

        for j, col in enumerate(level_cols):
            record[col] = parts[j] if j < len(parts) else ''

        records.append(record)
        current_id_video += 1

    df_new = pd.DataFrame(records)

    # Garante que todas as colunas existentes sejam preservadas
    all_columns = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
    df_old = df_old.reindex(columns=all_columns, fill_value='')
    df_new = df_new.reindex(columns=all_columns, fill_value='')

    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_csv(metadata_path, index=False)

    print(f"[✅] {len(df_new)} vídeos importados para {metadata_path}")

if __name__ == '__main__':
    main()
