# QuantumLib: Python Library for Quantum Computing

Welcome to QuantumLib, a Python library dedicated to quantum computing! This repository houses resources, documentation, and notebooks related to QuantumLib.

## Directory Structure

- [docs](docs/index.md): Markdown documentation files.
- [notebooks](notebooks): Jupyter notebooks categorized by language.
  - [English](notebooks/en/): English language notebooks.
    - [Circle Notation](notebooks/en/circle_notation.ipynb): Notebook explaining circle notation.
  - [Spanish](notebooks/es/): Spanish language notebooks.
    - [Basic Algebra](notebooks/es/basic_algebra.ipynb): Notebook covering basic algebra.
    - [Bell States](notebooks/es/bell_states.ipynb): Notebook explaining Bell states.
- [src](src/quantum_lib): Source code directory for QuantumLib.
  - [quantum_lib](src/quantum_lib): Core library package.
    - [notations](src/quantum_lib/notations): Module for different quantum notations.
      - [__init__.py](src/quantum_lib/notations/__init__.py): Initialization file for the notations module.
      - [algebra.py](src/quantum_lib/notations/algebra.py): Module handling quantum algebra.
      - [exceptions.py](src/quantum_lib/notations/exceptions.py): Custom exceptions for the notations module.
      - [gates.py](src/quantum_lib/notations/gates.py): Module containing gate implementations.
      - [states.py](src/quantum_lib/notations/states.py): Module managing quantum states.
      - [utils.py](src/quantum_lib/notations/utils.py): Utility functions for quantum operations.
    - [__init__.py](src/quantum_lib/__init__.py): Initialization file for QuantumLib package.
- [.gitignore](.gitignore): Git ignore file.
- [LICENSE](LICENSE): License information for the repository.
- [README.md](README.md): Project overview and instructions.
- [requirements.txt](requirements.txt): File containing required dependencies.

Developers interested in quantum computing can explore QuantumLib's modules and functionalities to learn, experiment, and implement quantum algorithms. Contributions are encouraged and welcomed.
