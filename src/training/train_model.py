import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Treina uma CNN (FCN) para segmentação binária de vegetação."
    )
    parser.add_argument(
        "--rgb",
        required=True,
        type=str,
        help="Diretório contendo imagens RGB (blocos)."
    )
    parser.add_argument(
        "--groundtruth",
        required=True,
        type=str,
        help="Diretório contendo máscaras binárias (0/1)."
    )
    parser.add_argument(
        "--modelpath",
        required=True,
        type=str,
        help="Caminho onde o modelo treinado será salvo (.h5)."
    )
    return parser.parse_args()


def load_dataset(rgb_dir: Path, mask_dir: Path, image_size=(256, 256)):
    rgb_files = {p.name: p for p in rgb_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}
    mask_files = {p.name: p for p in mask_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]}

    # Encontra interseção dos nomes
    common_files = sorted(set(rgb_files.keys()) & set(mask_files.keys()))

    if not common_files:
        raise RuntimeError("Nenhum par correspondente entre imagens RGB e máscaras.")

    print(f"[INFO] Imagens RGB encontradas: {len(rgb_files)}")
    print(f"[INFO] Máscaras encontradas: {len(mask_files)}")
    print(f"[INFO] Pares válidos RGB/Máscara: {len(common_files)}")

    X = []
    y = []

    for filename in common_files:
        rgb_path = rgb_files[filename]
        mask_path = mask_files[filename]

        img = Image.open(rgb_path).convert("RGB").resize(image_size)
        mask = Image.open(mask_path).convert("L").resize(image_size)

        img_np = np.array(img, dtype=np.float32) / 255.0  # normaliza RGB
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = (mask_np > 0).astype(np.float32)        # garante 0 ou 1
        mask_np = np.expand_dims(mask_np, axis=-1)       # (H, W, 1)

        X.append(img_np)
        y.append(mask_np)

    return np.array(X), np.array(y)



def build_fcn(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)

    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)

    # Saída pixel-a-pixel
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model


def train_model(X, y, model_path: Path):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_fcn(input_shape=X.shape[1:])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("[INFO] Iniciando treinamento...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=8
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"[DONE] Modelo salvo em: {model_path.resolve()}")


def main():
    args = parse_arguments()

    rgb_dir = Path(args.rgb)
    mask_dir = Path(args.groundtruth)
    model_path = Path(args.modelpath)

    X, y = load_dataset(rgb_dir, mask_dir)
    train_model(X, y, model_path)


if __name__ == "__main__":
    main()
