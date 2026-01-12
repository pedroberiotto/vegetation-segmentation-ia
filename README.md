# Vegetation Segmentation – Desafio de IA

Este repositório implementa um pipeline completo para **segmentação de vegetação em imagens aéreas/ortomosaicos**, seguindo as 4 etapas solicitadas no desafio:

1. Quebra do ortomosaico em blocos (tiles)
2. Geração de dataset (binarização)
3. Treinamento de uma Rede Neural (CNN – FCN)
4. Inferência do modelo em imagens não vistas

---

## 📁 Estrutura do Repositório

```
vegetation-segmentation/
│
├── data/
│   ├── raw/                 # Ortomosaicos originais (TIFF)
│   ├── tiles/               # Blocos gerados (Etapa 1)
│   ├── masks/               # Máscaras binárias (Etapa 2)
│   ├── inference_inputs/    # Imagens externas para inferência
│   └── inference_outputs/   # Resultados da inferência
│
├── models/                  # Modelos treinados (.h5)
│
├── src/
│   ├── preprocessing/
│   │   └── divide_orthomosaic.py
│   ├── dataset/
│   │   └── binarize_images.py
│   ├── training/
│   │   └── train_model.py
│   └── inference/
│       └── model_inference.py
│
├── scripts/
│   ├── validate_dataset.py
│   ├── count_mask_values.py
│   └── run_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## 📂 Observação sobre os Dados 

As pastas dentro de `data/` estão intencionalmente vazias neste repositório.

Devido às **restrições de tamanho do Git/GitHub para arquivos grandes (ex.: ortomosaicos e imagens TIFF)**, os dados de entrada e saída não foram versionados.

Para executar o pipeline corretamente, coloque o arquivo de ortomosaico (ex.: `orthomosaic.tif`) em `data/raw/`.
   
---

## ⚙️ Ambiente

### Criar ambiente virtual

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Instalar dependências

```bash
pip install -r requirements.txt
```
---

### 🧩 Etapa 1 — Quebra do Ortomosaico

```bash
python src/preprocessing/divide_orthomosaic.py \
  --input data/raw/orthomosaic.tif \
  --output data/tiles/ \
  --tile_size 512
```

> **Nota:** `--tile_size` é **opcional** (padrão: 512).

**Resultado:** imagens menores (tiles) em `data/tiles/`.

---

## 🌱 Etapa 2 — Geração de Dataset (Binarização)

Gera as máscaras de segmentação (ground truth) a partir dos tiles RGB.

* Pixels com vegetação → **1**
* Pixels sem vegetação → **0**

A binarização utiliza o índice **ExG (Excess Green Index)** com limiarização automática (Otsu).

**Script:** `src/dataset/binarize_images.py`

**Comando:**

```bash
python src/dataset/binarize_images.py \
  --input data/tiles/ \
  --output data/masks/
```

**Resultado:** máscaras em escala de cinza (0/1) em `data/masks/`.

---

## 🔎 Validação do Dataset

Antes do treinamento, o dataset é validado automaticamente.

### 1. Validação estrutural

Verifica:

* Correspondência 1–para–1 entre tiles e máscaras
* Dimensões iguais
* Máscaras em escala de cinza
* Valores apenas {0,1}

**Script:** `scripts/validate_dataset.py`

```bash
python scripts/validate_dataset.py \
  --rgb data/tiles/ \
  --masks data/masks/
```

### 2. Estatísticas do dataset

Conta quantos pixels pertencem a cada classe (0 e 1) para análise de balanceamento.

**Script:** `scripts/count_mask_values.py`

```bash
python scripts/count_mask_values.py \
  --masks data/masks/
```

---

## 🤖 Etapa 3 — Treinamento do Modelo

Foi utilizada uma **CNN do tipo FCN (Fully Convolutional Network)** para segmentação binária:

* Arquitetura encoder–decoder simples
* Saída pixel a pixel
* Função de perda: `binary_crossentropy`

**Script:** `src/training/train_model.py`

**Comando (conforme enunciado):**

```bash
python src/training/train_model.py \
  --rgb data/tiles/ \
  --groundtruth data/masks/ \
  --modelpath models/vegetation_model.h5
```

**Resultado:** modelo treinado salvo em `models/vegetation_model.h5`.

---

## 🔮 Etapa 4 — Inferência do Modelo

Aplica o modelo treinado em uma **imagem RGB não utilizada no treinamento**, avaliando a capacidade de generalização.

A interface de execução segue **exatamente o comando solicitado no enunciado**:

```bash
python model_inference.py --rgb </path/to/image.png> --modelpath </path/to/model.h5> --output </path/to/segmented/image.png>
```

No projeto:

**Script:** `src/inference/model_inference.py`

```bash
python src/inference/model_inference.py \
  --rgb data/inference_inputs/teste.jpg \
  --modelpath models/vegetation_model.h5 \
  --output data/inference_outputs/teste_segmentado.png
```

**Resultado:** imagem segmentada em escala de cinza:

* **0 (preto)** → não vegetação
* **255 (branco)** → vegetação

---

## 🚀 Pipeline Automatizada

Além da execução individual de cada etapa, foi implementada uma pipeline que executa todo o fluxo de ponta a ponta:

1. Limpeza de diretórios (`tiles`, `masks`, `inference_outputs`)
2. Divisão do ortomosaico
3. Binarização
4. Validação do dataset
5. Estatísticas de classes
6. Treinamento do modelo
7. Inferência

**Script:** `scripts/run_pipeline.py`

**Comando:**

```bash
python scripts/run_pipeline.py \
  --orthomosaic data/raw/orthomosaic.tif \
  --inference_image data/inference_inputs/teste.jpg \
  --tile_size 128 \
  --modelpath models/vegetation_model.h5
```

A pipeline **respeita integralmente a interface exigida na Etapa 4**, apenas automatizando sua execução.

---

## 📊 Generalização do Modelo

O modelo foi testado com imagens externas (capturadas por drones ou obtidas de bancos públicos), conforme sugerido no enunciado.

Observações típicas:

* Boa detecção em regiões com vegetação densa
* Limitações em áreas com vegetação esparsa ou baixo contraste
* Sensibilidade a artefatos gráficos (linhas, sombras)

Esses testes demonstram **capacidade de generalização**, bem como oportunidades de melhoria com dados multiespectrais ou rótulos manuais.
