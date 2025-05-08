import cv2
import mediapipe as mp
import pandas as pd
import os
import argparse

# Defina aqui as pastas de entrada e saída
VIDEO_FOLDER = "../../data/datasets/video"
OUTPUT_FOLDER = "../../data/datasets/keypoints"

def process_video_mediapipe(video_path, output_folder, density, keypoints_type):
    model_complexity = {"lite": 0, "full": 1, "heavy": 2}[density]

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo {video_path}")
            return

        data_normalized = []
        data_world = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                normalized_keypoints = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ]
                data_normalized.append([coord for point in normalized_keypoints for coord in point])

                if results.pose_world_landmarks:
                    world_keypoints = [
                        [lm.x, lm.y, lm.z, lm.visibility]
                        for lm in results.pose_world_landmarks.landmark
                    ]
                    data_world.append([coord for point in world_keypoints for coord in point])

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

            cv2.imshow("Pose Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

        columns = [f"{axis}_{i}" for i in range(33) for axis in ["x", "y", "z", "visibility"]]
        filename_base = os.path.splitext(os.path.basename(video_path))[0]

        if keypoints_type == "normalized" and data_normalized:
            pd.DataFrame(data_normalized, columns=columns).to_csv(
                os.path.join(output_folder, f"{filename_base}_mediapipe_{density}_normalized.csv"), index=False
            )
        elif keypoints_type == "landmark" and data_world:
            pd.DataFrame(data_world, columns=columns).to_csv(
                os.path.join(output_folder, f"{filename_base}_mediapipe_{density}_world.csv"), index=False
            )

def main():
    parser = argparse.ArgumentParser(description="Pose Estimation Tool")
    parser.add_argument("--model", choices=["mediapipe"], required=True, help="Modelo de pose a ser utilizado")
    parser.add_argument("--density", choices=["lite", "full", "heavy"], default="full", help="Complexidade do modelo (MediaPipe)")
    parser.add_argument("--keypoints", choices=["normalized", "landmark"], default="normalized", help="Tipo de pontos-chave a serem extraídos")
    args = parser.parse_args()

    # Verificar se a pasta de entrada existe
    if not os.path.exists(VIDEO_FOLDER):
        print(f"A pasta de vídeos '{VIDEO_FOLDER}' não existe!")
        return

    # Criar a pasta de saída se não existir
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Tela cheia
    cv2.namedWindow("Pose Detection", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for video_file in os.listdir(VIDEO_FOLDER):
        if video_file.endswith(".avi"):
            video_path = os.path.join(VIDEO_FOLDER, video_file)
            print(f"Processando {video_file}...")
            if args.model == "mediapipe":
                process_video_mediapipe(video_path, OUTPUT_FOLDER, args.density, args.keypoints)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
