"""
Dynamically creates a matrix for the GitHub Actions workflow that
runs the notebooks in the examples directory.
"""

import os
import json

# Dictionary to store custom timeouts for each notebook
notebook_timeouts = {
    "ADCP_Delft3D_TRTS_example.ipynb": 2400,
    "adcp_example.ipynb": 240,
    "adv_example.ipynb": 180,
    "cdip_example.ipynb": 420,
    "Delft3D_example.ipynb": 180,
    "directional_waves.ipynb": 180,
    "environmental_contours_example.ipynb": 240,
    "extreme_response_contour_example.ipynb": 240,
    "extreme_response_full_sea_state_example.ipynb": 240,
    "extreme_response_MLER_example.ipynb": 240,
    "loads_example.ipynb": 180,
    "metocean_example.ipynb": 180,
    "mooring_example.ipynb": 240,
    "PacWave_resource_characterization_example.ipynb": 240,
    "power_example.ipynb": 180,
    "qc_example.ipynb": 180,
    "river_example.ipynb": 180,
    "short_term_extremes_example.ipynb": 180,
    "strain_measurement_example.ipynb": 30,
    "SWAN_example.ipynb": 180,
    "tidal_example.ipynb": 180,
    "tidal_performance_example.ipynb": 180,
    "upcrossing_example.ipynb": 180,
    "wave_example.ipynb": 180,
    "wecsim_example.ipynb": 180,
    "WPTO_hindcast_example.ipynb": 180,
    "default": 60,  # Default timeout for other notebooks
}
notebooks = []
for root, dirs, files in os.walk("examples"):
    for file in files:
        if file.endswith(".ipynb"):
            notebooks.append(os.path.join(root, file))

# Generate the matrix configuration
matrix = {"include": []}
for notebook in notebooks:
    timeout = notebook_timeouts.get(
        os.path.basename(notebook), notebook_timeouts["default"]
    )

    matrix["include"].append({"notebook": notebook, "timeout": timeout})


# Print the matrix as a properly formatted JSON string
matrix_json = json.dumps(matrix)
print(f"matrix={matrix_json}")
