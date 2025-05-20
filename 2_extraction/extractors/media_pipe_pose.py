import cv2
import mediapipe as mp
import pandas as pd
import os
import sys
import argparse

DATA_VIDEOS = "../../data/1_videos"
DATA_KEYPOITNS = "../../data/2_keypoints"
VIDEO_LISTS = "../../lists/1_videos/"
METADATA_KEYPOTINS = "../../metadata/2_keypoints.csv"

def process_video(video_path, output_folder, model_name, model_complexity, show_visual, keypoint_type, video_metadata_row, extractor_name="mediapipe_pose", metadata_df=None):
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        cap = cv2.VideoCapture(video_path)

        data = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                if keypoint_type == "normalized":
                    keypoints = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in results.pose_landmarks.landmark
                    ]
                elif keypoint_type == "world" and results.pose_world_landmarks:
                    keypoints = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in results.pose_world_landmarks.landmark
                    ]
                else:
                    keypoints = None

                if keypoints is not None:
                    flattened = [coord for point in keypoints for coord in point]
                    data.append(flattened)
                    frame_count += 1  # só conta frames com keypoints detectados

                if show_visual:
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

            if show_visual:
                cv2.imshow("Pose Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

        columns = [f"{axis}_{i}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        if data:
            pd.DataFrame(data, columns=columns).to_csv(
                os.path.join(output_folder, f"{base_name}_{model_name}_{keypoint_type}.csv"), index=False
            )

        # Atualizar o dataframe de metadata com apenas uma linha por vídeo
        if metadata_df is not None:
            meta_row = video_metadata_row.copy()
            meta_row["extractor_name"] = extractor_name
            meta_row["model"] = model_name
            meta_row["keypoint_type"] = keypoint_type
            meta_row["number_of_frames"] = frame_count  # total frames com keypoints detectados

            # Adicionar essa linha ao dataframe metadata_df
            metadata_df = pd.concat([metadata_df, pd.DataFrame([meta_row])], ignore_index=True)

        return metadata_df


        cap.release()

        columns = [f"{axis}_{i}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        if data:
            # Salvar CSV dos keypoints
            pd.DataFrame(data, columns=columns).to_csv(
                os.path.join(output_folder, f"{base_name}_{model_name}_{keypoint_type}.csv"), index=False
            )

        # Atualizar o dataframe de metadata, se fornecido
        if metadata_df is not None and len(metadata_rows) > 0:
            metadata_df_new = pd.DataFrame(metadata_rows)
            # Concatenar com o dataframe existente
            metadata_df = pd.concat([metadata_df, metadata_df_new], ignore_index=True)

        return metadata_df
    



def main():
    parser = argparse.ArgumentParser(description="Extrair pontos-chave de vídeos com MediaPipe Pose.")
    parser.add_argument("list_name", help="Arquivo CSV com lista dos vídeos (primeira coluna, nomes sem extensão)")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="lite", help="Modelo MediaPipe a usar")
    parser.add_argument("--visual", action="store_true", help="Mostrar janela de visualização durante extração")
    parser.add_argument("--keypoint_type", choices=["normalized", "world"], default="normalized", help="Tipo de pontos-chave a salvar (padrão: normalized)")

    args = parser.parse_args()

    video_folder = DATA_VIDEOS
    keypoint_folder = DATA_KEYPOITNS
    os.makedirs(keypoint_folder, exist_ok=True)

    df = pd.read_csv(VIDEO_LISTS + args.list_name, header=None)
    video_names_to_process = set(df.iloc[:,0].astype(str))

    # Carregar metadata atual, ou criar vazio
    if os.path.exists(METADATA_KEYPOTINS):
        metadata_df = pd.read_csv(METADATA_KEYPOTINS)
    else:
        metadata_df = pd.DataFrame()

    models = {
        "lite": 0,
        "full": 1,
        "heavy": 2,
    }
    model_name = args.model
    model_complexity = models[model_name]

    for video_file in os.listdir(video_folder):
        base_name = os.path.splitext(video_file)[0]
        if base_name in video_names_to_process:
            video_path = os.path.join(video_folder, video_file)

            if not video_file.lower().endswith(('.avi', '.mp4', '.mkv', '.mov', '.flv', '.wmv', '.mpeg', '.mpg')):
                continue

            if args.visual:
                cv2.namedWindow("Pose Detection", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            print(f"[INFO] Processando vídeo {video_file} com modelo {model_name} e keypoint_type {args.keypoint_type}...")

            # Pega a linha do CSV com os dados completos para esse vídeo (sem extensão)
            video_metadata_row = df[df.iloc[:,0] == base_name].iloc[0].to_dict()

            metadata_df = process_video(
                video_path,
                keypoint_folder,
                model_name,
                model_complexity,
                args.visual,
                args.keypoint_type,
                video_metadata_row,
                extractor_name="mediapipe_pose",
                metadata_df=metadata_df
            )

    # Salvar metadata atualizado
    metadata_df.to_csv(METADATA_KEYPOTINS, index=False)

    if args.visual:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
