# Contributing to QCML

First off, thank you for considering contributing to the **QCML** project! Your involvement is key to improving this project, and we’re excited to collaborate with you.

## Table of Contents
- [How to Contribute](#how-to-contribute)
- [Code of Conduct](#code-of-conduct)
- [Bug Reports](#bug-reports)
- [Feature Requests](#feature-requests)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

---

## How to Contribute

1. **Fork the repository**:  
   Use the "Fork" button at the top right of the repository page on GitHub to create a copy of the project in your own GitHub account.

2. **Clone your fork**:  
   ```bash
   git clone https://github.com/albertnieto/qcml.git
   cd qcml
   ```

3. **Create a new branch**:  
   Create a branch for your contribution.
   ```bash
   git checkout -b my-feature-branch
   ```

4. **Make your changes**:  
   Ensure your changes follow the coding standards and are accompanied by relevant tests.

5. **Commit your changes**:  
   Write clear and concise commit messages.
   ```bash
   git commit -m "Add feature X"
   ```

6. **Push your branch**:  
   Push to your fork on GitHub.
   ```bash
   git push origin my-feature-branch
   ```

7. **Submit a pull request**:  
   Open a pull request from your branch to the main `qcml` repository. Follow the pull request template and describe your changes in detail.

---

## Code of Conduct

We adhere to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Please read it carefully to understand the expected behavior when interacting with the project.

---

## Bug Reports

If you find a bug, please help us by reporting it via the [GitHub issues](https://github.com/albertnieto/qcml/issues). Before reporting, please:

1. **Search existing issues**: Your issue may already be reported.
2. **Include details**: Describe the bug, expected behavior, and steps to reproduce it.
3. **Provide logs and screenshots** (if applicable): Help us understand the issue better by attaching logs or screenshots.

---

## Feature Requests

We welcome suggestions for new features or improvements! To propose a new feature:

1. Open an issue with the tag `enhancement`.
2. Provide a detailed explanation of the feature and its benefits.
3. If possible, describe how it could be implemented.

---

## Coding Standards

Please adhere to the following coding standards:

- **Python Style**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- **Naming Conventions**: Use meaningful names for variables, functions, classes, etc.
- **Type Hints**: Include Python type hints where possible to improve code readability.
- **Docstrings**: Write clear and concise docstrings for all public modules, functions, and classes following [PEP 257](https://www.python.org/dev/peps/pep-0257/).

---

## Testing

Ensure that your contribution includes adequate tests:

- Add new tests in the `tests/` folder.
- Use `pytest` to run all tests.
  bash
  pytest
- If fixing a bug, write a regression test that fails before your fix and passes afterward.
- Coverage should not decrease—ensure tests cover all new functionality.

---

## Documentation

Documentation is an essential part of the QCML project. All public classes, functions, and modules must be well documented. The documentation is located in the `docs/` directory and can be built using:

bash
mkdocs serve

Please update the documentation accordingly if you add new features or make significant changes.

---

## Pull Request Process

To submit a pull request (PR):

1. Ensure all changes are committed and pushed to your fork.
2. Submit your PR to the `main` branch.
3. Your PR will be reviewed by a project maintainer. Please be patient, as reviews can take time.
4. Once approved, your branch will be merged into `main`.
5. If requested, please make any necessary changes based on the reviewer’s feedback.

When submitting your pull request:

- **Make sure all tests pass**: Run `pytest` to verify.
- **Keep it small**: Large pull requests are harder to review. Try to break down large changes into smaller, self-contained contributions.
- **Update documentation**: If your change modifies functionality, ensure the documentation reflects those changes.

---

Thank you for contributing to QCML! We look forward to seeing your work improve this project.
