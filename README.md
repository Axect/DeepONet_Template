# DeepONet Template Library

This is a template library for conducting research on DeepONet, a deep learning architecture for learning operators and solving partial differential equations. The library provides a modular and extensible framework for defining, training, and evaluating DeepONet models.

## Directory Structure

The template library has the following directory structure:

```
./
├── README.md
├── analyze.py
├── deeponet
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── freeze.sh
├── kan.sh
├── requirements.txt
└── run.py
```

- `README.md`: This file, providing an overview and documentation of the library.
- `analyze.py`: A script for analyzing trained models and generating visualizations.
- `deeponet/`: The main package containing the core components of the library.
  - `__init__.py`: Package initialization file.
  - `data.py`: Module for loading and preprocessing data.
  - `model.py`: Module defining the DeepONet model architectures.
  - `train.py`: Module for training DeepONet models.
  - `utils.py`: Module containing utility functions and classes.
- `freeze.sh`: A shell script for freezing the library's dependencies into `requirements.txt`.
- `kan.sh`: A shell script for cloning and setting up the Kolmogorov-Arnold Network (KAN) repository.
- `requirements.txt`: A file listing the library's dependencies.
- `run.py`: The main script for running experiments and training models.

## Prerequisites

- Python 3.7 or higher
- Git
- [uv](https://github.com/astral-sh/uv)

## Installation

To install the library and its dependencies, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Axect/DeepONet_Template
   ```

2. Navigate to the library directory:
   ```
   cd DeepONet_Template
   ```

3. Set up a virtual environment via uv:
   ```
   uv venv
   uv pip sync requirements.txt
   source .venv/bin/activate
   ```

4. (Optional) Run the `kan.sh` script to download efficient-kan (Kolmogorov-Arnold Network):
   ```
   sh kan.sh
   ```

## Usage

To use the DeepONet template library, follow these steps:

1. Prepare your data:
   - Organize your data into the appropriate format required by the library.
   - Modify the `data.py` module to load and preprocess your data.

2. Define your model:
   - Choose the appropriate DeepONet model architecture from the `model.py` module or create a new one.
   - Modify the model architecture and hyperparameters as needed.

3. Train your model:
   - Use the `run.py` script to train your model.
   - Adjust the training parameters and hyperparameters in the `run.py` script.

4. Analyze and visualize results:
   - Use the `analyze.py` script to analyze the trained models and generate visualizations.
   - Customize the analysis and visualization code in `analyze.py` based on your requirements.

5. (Optional) Freeze the dependencies:
   - If you make changes to the library's dependencies, run the `freeze.sh` script to update the `requirements.txt` file.

## Contributing

Contributions to the DeepONet template library are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This template library was inspired by various research papers and implementations of [DeepONet](https://github.com/lululxvi/deeponet).
We would like to acknowledge the contributors and researchers who have made significant contributions to the field of DeepONet and related areas.

