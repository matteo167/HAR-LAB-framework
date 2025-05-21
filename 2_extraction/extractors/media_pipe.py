import os
import cv2
import argparse
import pandas as pd
import mediapipe as mp

# Constantes de diretórios e arquivos
DATA_VIDEOS = "../../data/1_videos"
DATA_KEYPOINTS = "../../data/2_keypoints"
VIDEO_LISTS = "../../lists/1_videos/"
METADATA_KEYPOINTS = "../../metadata/2_keypoints.csv"
METADATA_VIDEO = "../../metadata/1_keypoints.csv"

# Função principal de extração
def process_video(
    video_path, output_folder, model_name, model_complexity,
    show_visual, keypoint_type, video_metadata_row,
    extractor_name="mediapipe_pose", metadata_df=None
):
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
                # Extrair pontos-chave
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
                    frame_count += 1  # apenas frames com keypoints

                if show_visual:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                    )

            if show_visual:
                cv2.imshow("Pose Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()

        # Salvar CSV dos keypoints
        if data:
            columns = [f"{axis}_{i}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_{model_name}_{keypoint_type}.csv")
            pd.DataFrame(data, columns=columns).to_csv(output_path, index=False)

        # Atualizar metadados
        if metadata_df is not None:
            meta_row = video_metadata_row.copy()
            meta_row.update({
                "extractor_name": extractor_name,
                "model": model_name,
                "keypoint_type": keypoint_type,
                "number_of_frames": frame_count
            })
            metadata_df = pd.concat([metadata_df, pd.DataFrame([meta_row])], ignore_index=True)

        return metadata_df

def main():
    parser = argparse.ArgumentParser(description="Extrair pontos-chave de vídeos com MediaPipe Pose.")
    parser.add_argument("list_name", help="CSV com lista de vídeos (sem extensão)")
    parser.add_argument("--model", choices=["lite", "full", "heavy"], default="lite", help="Modelo MediaPipe")
    parser.add_argument("--visual", action="store_true", help="Mostrar janela de visualização")
    parser.add_argument("--keypoint_type", choices=["normalized", "world"], default="normalized", help="Tipo de keypoints")

    args = parser.parse_args()

    # Criar diretório de saída se não existir
    os.makedirs(DATA_KEYPOINTS, exist_ok=True)

    # Vídeos a processar
    video_names = pd.read_csv(os.path.join(VIDEO_LISTS, args.list_name), header=None)
    video_names_set = set(video_names.iloc[:, 0].astype(str))

    # Carregar metadados existentes ou iniciar um novo
    metadata_df = pd.read_csv(METADATA_KEYPOINTS) if os.path.exists(METADATA_KEYPOINTS) else pd.DataFrame()

    model_complexity_map = {"lite": 0, "full": 1, "heavy": 2}
    model_name = args.model
    model_complexity = model_complexity_map[model_name]

    # Processar cada vídeo
    for video_file in os.listdir(DATA_VIDEOS):
        base_name, ext = os.path.splitext(video_file)
        if base_name not in video_names_set:
            continue
        if ext.lower() not in ['.avi', '.mp4', '.mkv', '.mov', '.flv', '.wmv', '.mpeg', '.mpg']:
            continue

        video_path = os.path.join(DATA_VIDEOS, video_file)

        if args.visual:
            cv2.namedWindow("Pose Detection", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print(f"[INFO] Processando: {video_file} | Modelo: {model_name} | Keypoints: {args.keypoint_type}")

        video_metadata_row = video_names[video_names.iloc[:, 0] == base_name].iloc[0].to_dict()

        metadata_df = process_video(
            video_path=video_path,
            output_folder=DATA_KEYPOINTS,
            model_name=model_name,
            model_complexity=model_complexity,
            show_visual=args.visual,
            keypoint_type=args.keypoint_type,
            video_metadata_row=video_metadata_row,
            metadata_df=metadata_df
        )

    print(metadata_df)
    # Salvar metadados atualizados
    metadata_df.to_csv(METADATA_KEYPOINTS, index=False)

    if args.visual:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
