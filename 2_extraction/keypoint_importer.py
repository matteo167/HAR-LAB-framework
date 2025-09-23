# generic importer for testing
# Example: python3 importer.py teste /home/matteo/matteo/datasets/Teste --folder --copy

import argparse
import shutil
from pathlib import Path
import pandas as pd

KEYPOINT_EXTENSIONS = {'.csv'}
METADATA_KEYPOINTS = Path("../metadata/2_keypoints.csv")
DATA_KEYPOINTS = Path("../data/2_keypoints")

def is_keypoint_file(path: Path) -> bool:
    return True

def read_metadata(csv_path: Path) -> pd.DataFrame:
    """Lê o CSV de metadados ou cria um DataFrame vazio com as colunas padrão."""
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=['id_keypoint', 'dataset', 'name', 'extension'])

def generate_level_columns(max_level: int, prefix: str) -> list[str]:
    """Gera nomes de colunas para cada nível de subpasta."""
    return [f"{prefix}_level_{i}" for i in range(1, max_level + 1)]

def process_keypoints(base_path: Path) -> tuple[list[Path], list[list[str]], int]:
    """Retorna lista de arquivos de keypoints, partes do caminho e profundidade máxima."""
    keypoint_files = [p for p in base_path.rglob('*') if p.is_file() and is_keypoint_file(p)]
    paths_parts = [list(p.relative_to(base_path).parts[:-1]) for p in keypoint_files]
    max_depth = max((len(parts) for parts in paths_parts), default=0)
    return keypoint_files, paths_parts, max_depth

def main():
    parser = argparse.ArgumentParser(description='Importa arquivos de keypoints e atualiza o CSV de metadados.')
    parser.add_argument('dataset', help='Nome do dataset')
    parser.add_argument('directory', help='Caminho do diretório com os arquivos de keypoints')
    parser.add_argument('--folder', action='store_true', help='Incluir níveis de subpastas como colunas')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--move', action='store_true', help='Move os arquivos')
    group.add_argument('--copy', action='store_true', help='Copia os arquivos')
    args = parser.parse_args()

    dataset = args.dataset
    base_path = Path(args.directory).resolve()
    metadata_path = METADATA_KEYPOINTS.resolve()
    dest_path = DATA_KEYPOINTS.resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Lê o CSV antigo e define o próximo id_keypoint disponível
    df_old = read_metadata(metadata_path)
    current_id_keypoint = df_old['id_keypoint'].astype(int).max() + 1 if not df_old.empty else 1

    keypoint_files, paths_parts, max_depth = process_keypoints(base_path)
    level_cols = generate_level_columns(max_depth, dataset) if args.folder else []
    base_cols = ['id_keypoint', 'dataset', 'name', 'extension']

    records: list[dict] = []
    for i, filepath in enumerate(keypoint_files):
        parts = paths_parts[i]
        ext = filepath.suffix.lower()
        new_name = f"{current_id_keypoint}{ext}"
        dest_file = dest_path / new_name

        if args.move:
            shutil.move(str(filepath), str(dest_file))
        else:
            shutil.copy2(str(filepath), str(dest_file))

        record = {
            'id_keypoint': current_id_keypoint,
            'dataset': dataset,
            'name': filepath.name,
            'extension': ext
        }

        for j, col in enumerate(level_cols):
            record[col] = parts[j] if j < len(parts) else ''

        records.append(record)
        current_id_keypoint += 1

    df_new = pd.DataFrame(records)

    # Garante que todas as colunas existentes sejam preservadas
    all_columns = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
    df_old = df_old.reindex(columns=all_columns, fill_value='')
    df_new = df_new.reindex(columns=all_columns, fill_value='')

    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined.to_csv(metadata_path, index=False)

    print(f"[✅] {len(df_new)} arquivos de keypoints importados para {metadata_path}")

if __name__ == '__main__':
    main()