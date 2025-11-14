import argparse
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time smile detection using a trained CNN model.")
    parser.add_argument("model", type=str, help="Path to the trained model checkpoint (e.g., smile_cnn_genki4k.pt)")
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("H", "W"), help="Input size expected by the model")
    parser.add_argument("--cascade", type=str, default=None, help="Path to a Haar cascade XML file (defaults to OpenCV's frontal face cascade)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Smile probability threshold")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open")
    parser.add_argument("--no-cuda", action="store_true", help="Force CPU inference")
    return parser.parse_args()


class SmileCNN(nn.Module):
    def __init__(self, image_size: Tuple[int, int]) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        dummy_input = torch.zeros(1, 3, *image_size)
        flatten_dim = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def load_model(model_path: str, image_size: Tuple[int, int], device: torch.device) -> SmileCNN:
    model = SmileCNN(image_size)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_face_detector(cascade_path: Optional[str]) -> cv2.CascadeClassifier:
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.isfile(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    return detector


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = load_model(args.model, tuple(args.image_size), device)
    detector = get_face_detector(args.cascade)

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(tuple(args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        for (x, y, w, h) in faces:
            face_rgb = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
            tensor = preprocess(face_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item()
            label = "Smiling" if prob >= args.threshold else "Not smiling"
            color = (0, 255, 0) if prob >= args.threshold else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{label}: {prob*100:.1f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Smile Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
