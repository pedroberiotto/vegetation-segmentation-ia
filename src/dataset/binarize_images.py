import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Binariza imagens RGB em blocos para gerar dataset de segmentação (0 = não vegetação, 1 = vegetação)."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Diretório contendo imagens RGB em blocos."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Diretório onde serão salvas as imagens binárias (escala de cinza)."
    )
    return parser.parse_args()


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def compute_exg(image: np.ndarray) -> np.ndarray:
    """
    Calcula o índice ExG (Excess Green):
    ExG = 2G - R - B
    """
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    return exg


def binarize_exg(exg: np.ndarray) -> np.ndarray:
    """
    Normaliza o ExG e aplica limiarização de Otsu.
    Retorna imagem binária com valores 0 e 1.
    """
    # Normaliza para 0–255
    exg_norm = cv2.normalize(exg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    exg_uint8 = exg_norm.astype(np.uint8)

    # Otsu threshold
    _, binary = cv2.threshold(
        exg_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binary


def process_image(image_path: Path) -> np.ndarray:
    """
    Carrega imagem RGB e gera máscara binária (0 ou 1).
    """
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    exg = compute_exg(img_np)
    mask = binarize_exg(exg)

    return mask


def save_mask(mask: np.ndarray, output_path: Path):
    """
    Salva máscara binária como imagem em escala de cinza (0 ou 1).
    """
    # Garante formato uint8
    mask_uint8 = mask.astype(np.uint8)
    img = Image.fromarray(mask_uint8, mode="L")
    img.save(output_path)


def binarize_directory(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    ensure_output_dir(output_dir)

    image_files = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if not image_files:
        raise RuntimeError("Nenhuma imagem encontrada no diretório de entrada.")

    print(f"[INFO] Imagens encontradas: {len(image_files)}")
    print(f"[INFO] Entrada: {input_dir.resolve()}")
    print(f"[INFO] Saída: {output_dir.resolve()}")

    for idx, img_path in enumerate(image_files):
        mask = process_image(img_path)

        output_path = output_dir / img_path.name
        save_mask(mask, output_path)

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"[INFO] Processadas {idx + 1}/{len(image_files)} imagens...")

    print(f"[DONE] Dataset binário gerado com sucesso em: {output_dir.resolve()}")


def main():
    args = parse_arguments()
    binarize_directory(
        input_dir=Path(args.input),
        output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()
