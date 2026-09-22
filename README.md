# SIBNet: A Parameter-Efficient Skip–Inception–BSConv Network

SIBNet is a lightweight convolutional architecture that combines residual (skip)
connections, Inception-style multi-branch feature extraction, and Blueprint
Separable Convolutions (BSConv). This repository contains the training
notebooks used to evaluate SIBNet against three baseline architectures
(MobileNetV1-BSConv, ResNet50, and InceptionV1) across four benchmark
datasets, along with two ablation studies that isolate the contribution of
each architectural component.

## Repository Structure

| Notebook | Description |
|---|---|
| `SIBNet-MNIST.ipynb` | Trains and evaluates SIBNet and all three baselines on MNIST. |
| `sibnet-cifar10.ipynb` | Trains and evaluates SIBNet and all three baselines on CIFAR-10. |
| `sibnet-cifar100.ipynb` | Trains and evaluates SIBNet and all three baselines on CIFAR-100. |
| `ablation-variant-convs.ipynb` | **Ablation 1:** compares 7 convolution operators (dense, separable, grouped, MixConv, asymmetric, Ghost, BSConv) inside the Inception branches, on CIFAR-100. |
| `02-component-wise-ablation.ipynb` | **Ablation 2:** removes each architectural component (Inception branching, BSConv, skip connections) individually and in combination, on CIFAR-100. |

> **Note:** the results below also include Fashion-MNIST, but no
> `SIBNet-FashionMNIST.ipynb` is listed among the files provided for this
> README. If that notebook exists, add it to the table above (and the repo)
> so the results and code stay in sync — otherwise, consider removing the
> Fashion-MNIST row from the results table below.

## Results

SIBNet consistently achieves the highest accuracy, precision, recall, and
F1-score across all four datasets, with the largest margin over baselines on
the most difficult dataset, CIFAR-100.

| Dataset | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|---|---|---|---|---|---|
| MNIST | MobileNetV1-BSConv | 99.26 | 99.26 | 99.25 | 99.26 |
| MNIST | ResNet50 | 99.52 | 99.52 | 99.51 | 99.52 |
| MNIST | InceptionV1 | 99.64 | 99.65 | 99.63 | 99.64 |
| MNIST | **SIBNet (Proposed)** | **99.68** | **99.68** | **99.67** | **99.68** |
| Fashion-MNIST | MobileNetV1-BSConv | 91.72 | 91.69 | 91.72 | 91.69 |
| Fashion-MNIST | ResNet50 | 92.61 | 92.60 | 92.61 | 92.60 |
| Fashion-MNIST | InceptionV1 | 93.58 | 93.57 | 93.58 | 93.57 |
| Fashion-MNIST | **SIBNet (Proposed)** | **94.63** | **94.60** | **94.63** | **94.61** |
| CIFAR-10 | MobileNetV1-BSConv | 79.07 | 78.92 | 79.07 | 78.95 |
| CIFAR-10 | ResNet50 | 86.45 | 86.45 | 86.45 | 86.44 |
| CIFAR-10 | InceptionV1 | 90.48 | 90.51 | 90.48 | 90.48 |
| CIFAR-10 | **SIBNet (Proposed)** | **93.00** | **93.01** | **93.00** | **93.00** |
| CIFAR-100 | MobileNetV1-BSConv | 44.93 | 44.84 | 44.93 | 44.53 |
| CIFAR-100 | ResNet50 | 58.69 | 58.71 | 58.69 | 58.57 |
| CIFAR-100 | InceptionV1 | 64.46 | 64.75 | 64.46 | 64.46 |
| CIFAR-100 | **SIBNet (Proposed)** | **72.62** | **72.93** | **72.62** | **72.64** |

### Parameter Efficiency

SIBNet achieves the above results with a fraction of the parameters of the
larger baselines:

| Model | Parameters |
|---|---|
| MobileNetV1-BSConv | ~0.22–0.24M |
| **SIBNet (Proposed)** | **~1.31–1.33M** |
| InceptionV1 | ~5.61–5.70M |
| ResNet50 | ~23.53–23.71M |

SIBNet uses roughly **18× fewer parameters than ResNet50** and **4–5× fewer
than InceptionV1**, while matching or exceeding both in accuracy on every
dataset tested. Parameter counts vary slightly by dataset due to differing
input channels (grayscale vs. RGB) and output classes (10 vs. 100).

## Ablation Studies

### Ablation 1 — Convolution Operator (`ablation-variant-convs.ipynb`)

Compares seven convolution operators used inside the Inception branches,
with all other components held fixed, trained for 30 epochs on CIFAR-100.

| Convolution Type | Params (M) | Best Val. Acc. (%) | Final Val. Acc. (%) | Test Acc. (%) | Training Time (s) |
|---|---|---|---|---|---|
| Regular (dense) | 3.74 | 69.58 | 69.24 | 69.55 | 3650.3 |
| Ghost | 2.46 | 69.44 | 69.26 | 69.06 | 2943.3 |
| Grouped (ResNeXt-style) | 1.48 | 69.30 | 69.30 | 68.54 | 2884.8 |
| Separable (MobileNet-style) | 1.37 | 69.01 | 69.01 | 68.86 | 2172.1 |
| **BSConv (adopted)** | **1.33** | 68.91 | 68.59 | 68.08 | 1632.5 |
| Asymmetric (factorized) | 1.94 | 68.53 | 68.35 | 68.35 | 2423.0 |
| MixConv | 1.45 | 66.87 | 66.87 | 67.66 | 3018.9 |

BSConv was selected for the final model: it retains 99% of the dense
convolution's validation accuracy while using 64.4% fewer parameters and
55.3% less training time.

### Ablation 2 — Component-Wise (`02-component-wise-ablation.ipynb`)

Removes each of the three architectural components (skip connections,
Inception branching, BSConv) individually and in every combination, also on
CIFAR-100.

| Ablation | Params | Best Val Acc | Final Val Acc | Test Acc | Time (s) |
|---|---|---|---|---|---|
| skip_only | 3,391,140 | 0.7065 | 0.7057 | 0.7051 | 977.9 |
| inception_skip | 3,738,788 | 0.7033 | 0.7008 | 0.6974 | 3636.5 |
| **full_model** | **1,330,980** | 0.6828 | 0.6818 | 0.6785 | 1590.1 |
| bsconv_skip | 986,020 | 0.6681 | 0.6622 | 0.6644 | 929.0 |
| bsconv_only | 944,292 | 0.6193 | 0.6193 | 0.6210 | 852.5 |
| inception_bsconv | 1,289,252 | 0.5856 | 0.5854 | 0.5883 | 1517.6 |
| inception_only | 3,697,060 | 0.5255 | 0.5186 | 0.5248 | 3533.3 |

Replacing standard convolutions with BSConv reduces backbone parameters by
71–72% (relative to `skip_only`) at a modest accuracy cost, and each
component contributes measurably to the full model's final performance.

## Requirements

```
torch
torchvision
numpy
matplotlib
```

Install with:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

1. Open the desired notebook in Jupyter, Google Colab, or Kaggle.
2. Run the setup cells (imports, model definitions, data loading) from top
   to bottom.
3. For the dataset notebooks (`SIBNet-MNIST.ipynb`, `sibnet-cifar10.ipynb`,
   `sibnet-cifar100.ipynb`), run each model's training cell in turn; final
   metrics and comparison plots are generated at the end.
4. For the ablation notebooks, run the shared setup section once, then run
   each ablation configuration's cell individually (they can be run across
   multiple sessions — results are cached to disk per config), and finish
   with the aggregation/plotting cell once all configs have completed.

## Citation

If you use this code or these results, please cite:

```bibtex
@article{sibnet,
  title   = {SIBNet: A Parameter-Efficient Skip-Inception-BSConv Network for Image Classification},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

## License

TODO — add a license (e.g., MIT, Apache-2.0) before making this repository public.
