import os
import cv2
import pandas as pd
import mediapipe as mp
import argparse

# Argumento para nome da extração
parser = argparse.ArgumentParser(description="Extrair keypoints dos vídeos com MediaPipe.")
parser.add_argument("extraction", type=str, help="Nome da extração (ex: mediapipe_v1)")
args = parser.parse_args()
extraction_name = args.extraction

# Constantes de diretórios e arquivos
DATA_VIDEOS = "../../data/1_videos"
DATA_KEYPOINTS = "../../data/2_keypoints"
METADATA_KEYPOINTS = "../../metadata/2_keypoints.csv"
METADATA_VIDEO = "../../metadata/1_videos.csv"

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# Lê os metadados dos vídeos
video_metadata = pd.read_csv(METADATA_VIDEO)

# Garante que a pasta de keypoints existe
os.makedirs(DATA_KEYPOINTS, exist_ok=True)

# Define o próximo id_keypoint baseado nos arquivos existentes
existing_files = os.listdir(DATA_KEYPOINTS)
existing_ids = [int(f.split('.')[0]) for f in existing_files if f.endswith('.csv') and f.split('.')[0].isdigit()]
next_id = max(existing_ids, default=0) + 1

# Lista para armazenar os dados dos metadados de keypoints
keypoints_metadata = []

# Itera sobre os vídeos
for idx, row in video_metadata.iterrows():
    id_video = row["id_video"]
    extension = row["extension"].lstrip(".")
    video_filename = f"{id_video}.{extension}"
    video_path = os.path.join(DATA_VIDEOS, video_filename)
    
    if not os.path.exists(video_path):
        print(f"[AVISO] Vídeo não encontrado: {video_path}")
        continue

    print(f"[INFO] Processando vídeo: {video_filename}")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_list = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_keypoints = []
            for lm in landmarks:
                frame_keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints_list.append(frame_keypoints)

    cap.release()

    if keypoints_list:
        id_keypoint = f"{next_id:05d}"  # zero-padded (ex: 0001, 0002, ...)
        output_csv = os.path.join(DATA_KEYPOINTS, f"{id_keypoint}.csv")

        df_keypoints = pd.DataFrame(keypoints_list)
        df_keypoints.to_csv(output_csv, index=False)
        print(f"[INFO] Keypoints salvos em: {output_csv}")

        kp_entry = {}
        kp_entry["id_keypoint"] = id_keypoint
        kp_entry["extrator"] = "mediapipe"
        kp_entry["n_frames"] = frame_count
        kp_entry["extraction"] = extraction_name
        kp_entry.update(row.to_dict())
        print(kp_entry)
        keypoints_metadata.append(kp_entry)

        next_id += 1
    else:
        print(f"[AVISO] Nenhum keypoint extraído: {id_video}")

# Atualiza o CSV de metadados
df_metadata_new = pd.DataFrame(keypoints_metadata)

if os.path.exists(METADATA_KEYPOINTS):
    df_existing = pd.read_csv(METADATA_KEYPOINTS)
    df_combined = pd.concat([df_existing, df_metadata_new], ignore_index=True)
else:
    df_combined = df_metadata_new


df_combined.to_csv(METADATA_KEYPOINTS, index=False)
print(f"[INFO] Metadados dos keypoints atualizados em: {METADATA_KEYPOINTS}")












