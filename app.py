import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from functions import compiled  # Assuming this extracts the Sudoku from the image
from sudokuSolver import is_solvable, solve  # Assuming these solve the Sudoku

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                board = compiled(filepath)  # Extract Sudoku from image
                return render_template('edit.html', board=board)
            except:
                return render_template('Erro.html')
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    board = [[int(float(request.form[f'cell-{i}-{j}'])) for j in range(9)] for i in range(9)]
    if is_solvable(board):
        solve(board)
        return render_template('solution.html', board=board)
    else:
        return render_template('edit.html', board=board, error="The Sudoku puzzle is not solvable.")

if __name__ == '__main__':
    app.run(debug=True)
