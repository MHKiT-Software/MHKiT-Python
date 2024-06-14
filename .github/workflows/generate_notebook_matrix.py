"""
Dynamically creates a matrix for the GitHub Actions workflow that runs the notebooks in the examples directory.
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
    if notebook.endswith("metocean_example.ipynb") or notebook.endswith(
        "WPTO_hindcast_example.ipynb"
    ):
        matrix["include"].append({"notebook": notebook, "condition": "true"})
    else:
        matrix["include"].append({"notebook": notebook, "condition": "false"})

print(json.dumps(matrix))
