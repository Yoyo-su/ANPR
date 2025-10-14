# ANPR Project
An Automatic Number Plate Recognition (ANPR) system built using Python and various image processing libraries.

## 🔧 Tech Stack
- **Python 3.13 - Primary programming language**
    -   **Pytest - Test Driven Development (TDD)**
    -   **matplotlib - Image processing and visualization**
    -   **scikit-image - Image processing**
    -   **numpy - Numerical computations**
- **Github - Repository management, CI/CD (Github-Actions)**


## Architecture

The project is structured as follows:

```
ANPR/
│── src/
│   ├── main.py
│   ├── utils/
│   │   ├── localisation.py
│   │   ├── cca.py
├── tests/                  # Unit and integration tests
│   ├── data/               # Sample data or data schemas for tests
│   └── test*.py            # Unit and integration tests for python functions (pytest)
├── .gitignore              # Files not to be pushed to remote repository
├── Makefile                # Automated environment setup & configuration
├── README.md               # Project overview
└── requirements.txt        # Third party Python modules
```

## 🚀 Setup & Deployment

## Future Features
- Web-based User Interface for uploading images and displaying results.
- Support for multiple languages and character sets on number plates.
- Integration with a database to store recognized number plates and associated metadata.