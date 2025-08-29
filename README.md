## Fruits Fresh vs Rotten Classification (CNN)

A Convolutional Neural Network (CNN) project to classify fruit images (apples, bananas, oranges) as fresh or rotten. The project includes a prepared dataset, a Jupyter Notebook for end-to-end training and evaluation, and a configurable environment suitable for Windows.

### Dataset

The dataset is already organized in an ImageFolder-friendly layout:

```
dataset/
  train/
    freshapples/
    freshbanana/
    freshoranges/
    rottenapples/
    rottenbanana/
    rottenoranges/
  test/
    freshapples/
    freshbanana/
    freshoranges/
    rottenapples/
    rottenbanana/
    rottenoranges/
```

- Each subfolder contains PNG images for its class.
- You can easily plug this into PyTorch `ImageFolder` or Keras `flow_from_directory`.

### Requirements

- Python 3.9+ recommended
- Jupyter Notebook or JupyterLab
- Common ML libraries (depending on the notebook):
  - numpy, pandas, matplotlib, scikit-learn
  - One deep learning framework (the notebook should make it clear):
    - PyTorch: torch, torchvision
    - or TensorFlow/Keras: tensorflow

If you don't have an environment yet, a quick start (CPU):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install --upgrade pip
pip install numpy pandas matplotlib scikit-learn torch torchvision  # for PyTorch
# or: pip install numpy pandas matplotlib scikit-learn tensorflow   # for TensorFlow
pip install jupyter
```

### Getting Started

1) Clone and open the project

```bash
git clone <your-repo-url>
cd Fruits_fresh_and-rotten_Using_CNN
```

2) (Optional) Create/activate a virtual environment and install dependencies (see Requirements above).

3) Launch Jupyter and open the notebook

```bash
jupyter notebook
```

Then open `Fruits_fresh_and-rotten_Using_CNN.ipynb` and run cells top-to-bottom.

### Notebook Overview

The notebook typically covers:
- Data loading and augmentation from `dataset/train` and `dataset/test`
- Model definition (CNN backbone)
- Training loop with validation
- Evaluation: accuracy, confusion matrix, classification report
- Example predictions/visualizations
- Optional: saving/loading model weights

### Notes on Version Control

- A `.gitignore` is included to keep the repo clean. Large binary artifacts (checkpoints, logs) and environment folders are ignored by default.
- The `dataset/` entry is commented out in `.gitignore`. If you do not want to track the large image dataset in Git, uncomment `dataset/` in `.gitignore` or use Git LFS.
- Consider storing trained weights in a dedicated `models/` or `saved_models/` directory (already ignored).

### Reproducing Results

- Ensure the same dataset structure and similar package versions.
- For deterministic runs, set seeds in NumPy and the chosen DL framework.
- If using GPU, results may vary slightly vs CPU.

### Troubleshooting

- Out of memory (GPU): reduce batch size or image resolution; disable heavy augmentations.
- Slow training: ensure you enabled GPU if available; use `num_workers` > 0 in PyTorch `DataLoader`.
- Import errors: confirm your virtual environment is active and packages are installed.

### License

This project is licensed under the terms in `LICENSE`.

### Acknowledgements

- Thanks to the open-source ML ecosystem (PyTorch/TensorFlow, NumPy, scikit-learn, Matplotlib).
- Dataset folder structure follows common `ImageFolder` conventions for simplicity.