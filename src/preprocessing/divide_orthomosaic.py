import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Divide um ortomosaico TIFF em blocos menores e salva como PNG ou JPG."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Caminho para o arquivo de ortomosaico (.tif)."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Diretório onde os blocos serão salvos."
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=512,
        help="Tamanho (em pixels) de cada bloco quadrado. Padrão: 512."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Formato de saída das imagens. Padrão: png."
    )
    return parser.parse_args()


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image

    min_val = image.min()
    max_val = image.max()

    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    norm = (image - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)


def save_tile(tile: np.ndarray, output_path: Path):
    tile = normalize_to_uint8(tile)

    if tile.shape[2] == 1:
        img = Image.fromarray(tile[:, :, 0], mode="L")
    elif tile.shape[2] == 3:
        img = Image.fromarray(tile, mode="RGB")
    elif tile.shape[2] == 4:
        img = Image.fromarray(tile, mode="RGBA")
    else:
        raise ValueError(f"Número de canais não suportado: {tile.shape[2]}")

    img.save(output_path)


def split_orthomosaic(input_path: Path, output_dir: Path, tile_size: int, output_format: str):
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    ensure_output_dir(output_dir)

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        bands = src.count

        print(f"[INFO] Imagem: {input_path.name}")
        print(f"[INFO] Dimensões: {width} x {height} | Bandas: {bands}")
        print(f"[INFO] Tile size: {tile_size} | Formato: {output_format.upper()}")

        tile_id = 0

        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                win_width = min(tile_size, width - left)
                win_height = min(tile_size, height - top)

                window = Window(left, top, win_width, win_height)
                tile = src.read(window=window)
                tile = np.transpose(tile, (1, 2, 0))

                if tile.size == 0:
                    continue

                filename = f"tile_{tile_id:06d}.{output_format}"
                output_path = output_dir / filename

                save_tile(tile, output_path)
                tile_id += 1

        print(f"[DONE] Total de blocos gerados: {tile_id}")
        print(f"[DONE] Saída: {output_dir.resolve()}")


def main():
    args = parse_arguments()
    split_orthomosaic(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        tile_size=args.tile_size,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
