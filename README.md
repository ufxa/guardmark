# GuardMark: A Robust Watermarking and Fingerprinting Framework for Intellectual Property Protection of Fine-Tuned Large Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20TIFS-blue)](https://github.com/ufxa/guardmark)

## Abstract

GuardMark is a three-layer framework for intellectual property protection of fine-tuned LLMs, combining statistical token watermarking (WIA), behavioral fingerprinting (FVA), and weight-space gradient embedding (WIA+). We introduce the **Watermark Robustness Index (WRI)**, the first composite metric unifying detection probability, false-positive rate, and attack robustness.

**Key results:** 96.5% detection rate, 2.3% FP rate, WRI = 0.849, 3.2% overhead.

## Repository Structure

```
guardmark/
├── README.md
├── paper/
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── src/
│   ├── agents/
│   ├── metrics/
│   └── evaluation/
├── data/
│   ├── results/
│   └── processed/
├── notebooks/
├── scripts/
│   └── build.sh
└── LICENSE
```

## How to Reproduce

```bash
# Install dependencies
pip install numpy scipy pandas

# Run all experiments (no GPU required — parametric simulation)
python3 src/run_experiments.py --seed 42 --output-dir data/results/

# Build PDF
./scripts/build.sh
```

## Citation

```bibtex
@article{costa2025guardmark,
  author  = {Costa, Allan Douglas},
  title   = {GuardMark: A Robust Watermarking and Fingerprinting Framework
             for Intellectual Property Protection of Fine-Tuned Large
             Language Models},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025},
  doi     = {10.1109/TIFS.2025.XXXXXXX}
}
```

## License

MIT License — see [LICENSE](LICENSE).
