<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&fontColor=0047AB&text=MCQRNN&height=120&fontSize=90">
</div>

Monotone Composite Quantile Regression Neural Network (MCQRNN) implemented in both TensorFlow and PyTorch.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RektPunk/mcqrnn.git
cd mcqrnn
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the examples

```bash
uv run run_tensorflow.py
uv run run_torch.py
```

## Acknowledgments

This is an **unofficial** implementation based on:

```bibtex
@article{cannon2018non,
  title={Non-crossing nonlinear regression quantiles by monotone composite quantile regression neural network, with application to rainfall extremes},
  author={Cannon, Alex J},
  journal={Stochastic environmental research and risk assessment},
  volume={32},
  number={11},
  pages={3207--3225},
  year={2018},
  publisher={Springer}
}
```
