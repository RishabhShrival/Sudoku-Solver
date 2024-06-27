# Sudoku Solver from image

This repository contains a web application that allows users to upload an image of a Sudoku puzzle, extract the puzzle, edit it if necessary, and solve it. The application uses a machine learning model trained on the MNIST dataset with ResNet50 to recognize digits, and includes a solver for the Sudoku puzzle.

## Components

1. **TrainingModel.ipynb**: Jupyter notebook for training the digit recognition model using the MNIST dataset and ResNet50 architecture.
2. **functions.py**: Contains the functions for extracting Sudoku puzzles from images using the trained model.
3. **SudokuSolver.py**: Contains the logic for solving the Sudoku puzzle.
4. **app.py**: The Flask application that ties everything together and serves the web interface.
5. **templates/**: Contains the HTML templates and CSS for the web pages.
6. **uploads/**: Directory where uploaded images are stored temporarily.

## Setup and Installation

### Prerequisites

- Python 3.x
- Flask
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/sudoku-solver-webapp.git
   cd sudoku-solver-webapp
