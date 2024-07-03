"""
Dynamically creates a matrix for the GitHub Actions workflow that
runs the notebooks in the examples directory.
"""

import os
import json

notebooks = []
for root, dirs, files in os.walk("examples"):
    for file in files:
        if file.endswith(".ipynb"):
            notebooks.append(os.path.join(root, file))

# Generate the matrix configuration
matrix = {"include": []}
for notebook in notebooks:
    matrix["include"].append({"notebook": notebook})

# Print the matrix as a properly formatted JSON string
matrix_json = json.dumps(matrix)
print(f"matrix={matrix_json}")
