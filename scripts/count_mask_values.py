import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Conta quantos pixels são 0 e 1 em máscaras binárias."
    )
    parser.add_argument(
        "--masks",
        required=True,
        type=str,
        help="Diretório contendo as máscaras binárias (0/1)."
    )
    return parser.parse_args()


def count_pixels(mask_dir: Path):
    if not mask_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {mask_dir}")

    mask_files = sorted(
        [p for p in mask_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    )

    if not mask_files:
        raise RuntimeError("Nenhuma máscara encontrada no diretório.")

    total_zeros = 0
    total_ones = 0
    total_pixels = 0

    print(f"[INFO] Máscaras encontradas: {len(mask_files)}")

    for idx, mask_path in enumerate(mask_files):
        mask = Image.open(mask_path)
        mask_np = np.array(mask)

        zeros = np.sum(mask_np == 0)
        ones = np.sum(mask_np == 1)

        total_zeros += zeros
        total_ones += ones
        total_pixels += mask_np.size

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[INFO] Processadas {idx + 1}/{len(mask_files)} imagens...")

    percent_zeros = (total_zeros / total_pixels) * 100
    percent_ones = (total_ones / total_pixels) * 100

    print("\n===== RESULTADO FINAL =====")
    print(f"Total de pixels: {total_pixels}")
    print(f"Pixels classe 0 (não vegetação): {total_zeros} ({percent_zeros:.2f}%)")
    print(f"Pixels classe 1 (vegetação): {total_ones} ({percent_ones:.2f}%)")


def main():
    args = parse_arguments()
    count_pixels(Path(args.masks))


if __name__ == "__main__":
    main()
