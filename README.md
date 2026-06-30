# Deep Learning Approaches for Crop Yield Forecasting Using Sentinel-2 and Weather Data

This repository contains the code and experiments for my master's thesis:

> **Deep Learning Approaches for Crop Yield Forecasting Using Sentinel Images and Weather Data: Case Study on the CropNet Soybean Dataset**

We develop two complementary deep learning models for county-level soybean yield prediction under climate change:

- **GraphCrossFormer** – a compact, weather-only Transformer with a feature graph and SeasonMask phenological gating.
- **CNN-GRU (multi-modal)** – a model that fuses Sentinel-2 imagery and HRRR weather via gated fusion, spatial attention, and SeasonMask-modulated GRU.

Both models operate on the **CropNet** dataset (soybean subset) and aim to capture spatial, temporal, and multi-modal structure in a parameter-efficient way.
## Architectures

### GraphCrossFormer (weather-only)

GraphCrossFormer is a county-level, weather-only model designed for settings where satellite imagery is unavailable or too expensive to process.

Key features:

- **SeasonMask**: A learnable, per-state Gaussian gate over the 186-day growing season that focuses the model on phenologically critical periods (e.g., flowering and grain filling).
- **Dual-path encoding**:
  - Temporal path: patch embedding + Transformer encoder over masked daily weather.
  - Feature-graph path: Graph Attention Network (GAT) over meteorological variables (temperature, VPD, precipitation, etc.).
- **Context injection**: Long-term climate, state, and county embeddings are combined into a context vector and injected via cross-attention and FiLM modulation.
- **Temporal and variable attention**: Separate attention pooling over time and over variables, followed by state-specific regression heads for yield.

This model uses only meteorological sequences (WRF-HRRR derived features) and county-level agronomic statistics.

### CNN-GRU (multi-modal Sentinel-2 + weather)

CNN-GRU is a multi-modal model that explicitly leverages Sentinel-2 imagery alongside HRRR weather data.

Pipeline:

1. **Image encoder (ResNet-style CNN)**  
   - Each Sentinel-2 tile (per county, per acquisition date) is processed by a residual CNN backbone with batch normalization and ReLU activations.
   - Global average pooling yields a compact visual embedding per grid cell.

2. **Weather encoder (1D CNN)**  
   - 28-day sliding windows of HRRR weather per grid cell are processed by a small 1D CNN.
   - Global average pooling yields a weather embedding per grid cell.

3. **Gated multi-modal fusion**  
   - A learned gate `g` (sigmoid MLP) mixes image and weather:
     \[
     v_{\text{fus}} = g \odot v_{\text{img}} + (1-g) \odot v_{\text{w}}
     \]
   - Allows the model to down-weight imagery when clouds or artefacts dominate, and up-weight meteorology when imagery is unreliable.

4. **Spatial attention over grids**  
   - An attention mechanism scores each grid cell and aggregates them into a county-level vector, focusing on cropland grids and suppressing non-crop areas.

5. **SeasonMask + GRU**  
   - SeasonMask is applied on the county-level temporal sequence, emphasising the critical phenological window.
   - A 2-layer GRU aggregates fused representations over all Sentinel-2 acquisition dates.

6. **Climate, year, and county embeddings + prediction head**  
   - Long-term climate (multi-year monthly means), year embeddings, and county-level yield statistics are concatenated with the GRU output.
   - A small MLP maps this concatenated vector to normalised yield, then denormalises to bu/acre.

GraphCrossFormer and CNN-GRU are designed to be **complementary**: the former focuses on structured meteorological relationships, the latter adds explicit spatial heterogeneity from imagery.
## Dataset

We use the **CropNet** dataset, an open, large-scale multi-modal dataset for climate-change-aware crop yield prediction, released by Lin et al. and available on Hugging Face.[web:51][web:57]

- Hugging Face dataset: https://huggingface.co/datasets/CropNet/CropNet [web:51]
- Official CropNet / MMST-ViT code: https://github.com/fudong03/MMST-ViT [web:52]

CropNet provides, for 2017–2022:

- **Sentinel-2 Imagery** – multispectral tiles partitioned into fine-grained 9×9 km grids.
- **WRF-HRRR Computed Dataset** – high-resolution meteorological variables (e.g., temperature, precipitation, humidity, VPD).
- **USDA Crop Dataset** – county-level yields for major crops (corn, cotton, soybean, winter wheat, etc.).[web:52]

### Subset used in this project

In this repo, we focus on:

- **Crop**: Soybean.
- **Spatial scope**: Five US states from the CropNet soybean subset (see `config.py` for the exact list).
- **Temporal scope**:
  - Training: soybean county-years from 2017–2020.
  - Validation: soybean county-years from 2020–2021.
  - Test: soybean county-years from 2021–2022 (327 test counties, as reported in the thesis).

Data files (local names):

- `soybean_train.json` – preprocessed CropNet soybean training set.
- `soybean_test.json` – preprocessed CropNet soybean test set.
- `USDA_Soybean_County_2021-2.csv` – county-level soybean yields for the test period.
- `HRRR_*.csv` – HRRR-derived weather features aligned with CropNet counties.

Because these files are large and derived from the CropNet dataset, **they are not committed to the Git repository**. Instead, we provide:

- Small samples in `data/samples/`.


## Compute environments

Training and experiments were run on two different compute platforms to demonstrate portability:

### Multi-modal CNN-GRU (Sentinel-2 + weather) – Vast.ai RTX 6000

- Platform: **Vast.ai** rented GPU server.
- GPU: **NVIDIA RTX 6000 (Pro)**.
- Typical configuration:
  - CUDA-compatible PyTorch.
  - 32–48 GB GPU memory.
  - Batch sizes and image resolution tuned to avoid out-of-memory (see `experiments/configs/cnn_gru_default.yaml`).

The CNN-GRU model was trained end-to-end on this machine due to its higher memory requirements for multispectral Sentinel-2 tiles and multi-modal fusion.

### GraphCrossFormer (weather-only) – Google Colab

- Platform: **Google Colab**.
- GPU: T4 / A100 (depending on session availability).
- The weather-only GraphCrossFormer is lightweight (~1.15M parameters) and fits comfortably in Colab’s memory.
- Training scripts in `src/training/train_graphcrossformer.py` are configured for Colab:
  - Lower batch size.
  - Automatic mixed precision (optional).
  - Model checkpoints saved to `/content` or Google Drive.

This split mirrors a realistic deployment scenario: heavy multi-modal models on dedicated GPU cloud, and compact weather-only models on more constrained platforms like Colab.


    --config experiments/configs/graphcrossformer_default.yaml

python -m src.evaluation.grad_cam \
    --checkpoint experiments/checkpoints/cnn_gru_best.pt
```

Grad-CAM figures and training curves are saved under `assets/figures/` and `experiments/runs/`.
