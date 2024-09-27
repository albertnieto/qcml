# Copyright 2024 Albert Nieto

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from IPython.display import Math, Latex


def array_to_dirac_notation(superposition_array):
    """
    Convert a complex-valued array representing a quantum state in superposition
    to Dirac notation.

    Parameters:
    - superposition_array (numpy.ndarray): The complex-valued array representing
      the quantum state in superposition.

    Returns:
    str: The Dirac notation representation of the quantum state.
    """
    # Ensure the statevector is normalized
    superposition_array = superposition_array / np.linalg.norm(superposition_array)

    # Get the number of qubits
    num_qubits = int(np.log2(len(superposition_array)))

    # Initialize Dirac notation string
    dirac_notation = ""

    # Iterate through the array and add terms to the Dirac notation
    for i, amplitude in enumerate(superposition_array):
        # Skip negligible amplitudes
        if np.abs(amplitude) > 1e-10:
            # Convert the index to binary representation
            binary_rep = format(i, f"0{num_qubits}b")

            # Add the term to Dirac notation
            dirac_notation += f"{amplitude:.3f}|{binary_rep}⟩ + "

    # Remove the trailing " + " and return the result
    return dirac_notation[:-3]


def array_to_matrix_representation(array):
    """
    Convert a one-dimensional array to a column matrix representation.

    Parameters:
    - array (numpy.ndarray): The one-dimensional array to be converted.

    Returns:
    numpy.ndarray: The column matrix representation of the input array.
    """
    return array.reshape((len(array), 1))


def array_to_dirac_latex(array):
    """
    Generate LaTeX code for displaying the Dirac notation of a quantum state.

    Parameters:
    - array (numpy.ndarray): The complex-valued array representing the quantum state.

    Returns:
    Math: A Math object containing LaTeX code for displaying Dirac notation.
    """
    return Math(f"Dirac Notation:{array_to_dirac_notation(array)}")


def array_to_matrix_latex(array):
    """
    Generate LaTeX code for displaying the matrix representation of a one-dimensional array.

    Parameters:
    - array (numpy.ndarray): The one-dimensional array to be represented as a matrix.

    Returns:
    Math: A Math object containing LaTeX code for displaying the matrix representation.
    """
    matrix_representation = array_to_matrix_representation(array)
    latex = (
        "\\begin{bmatrix}\n"
        + "\\\\\n".join(map(str, matrix_representation.flatten()))
        + "\n\\end{bmatrix}"
    )
    return Math(f"Matrix Representation:{latex}")


def array_to_dirac_and_matrix_latex(array):
    """
    Generate LaTeX code for displaying both the matrix representation and Dirac notation
    of a quantum state.

    Parameters:
    - array (numpy.ndarray): The complex-valued array representing the quantum state.

    Returns:
    Latex: A Latex object containing LaTeX code for displaying both representations.
    """
    matrix_representation = array_to_matrix_representation(array)
    latex = (
        "Matrix representation\n\\begin{bmatrix}\n"
        + "\\\\\n".join(map(str, matrix_representation.flatten()))
        + "\n\\end{bmatrix}\n"
    )
    latex += f"Dirac Notation:\n{array_to_dirac_notation(array)}"
    return Latex(latex)


def matrix_to_latex(matrix, prefix=""):
    """
    Convert a NumPy matrix to its LaTeX representation.

    Parameters:
    - matrix (numpy.ndarray): The input matrix.
    - prefix (str): A string to be prepended to the LaTeX representation.

    Returns:
    IPython.display.Math: LaTeX representation of the matrix.
    """
    latex_code = f"{prefix}\\begin{{bmatrix}}\n"

    for row in matrix:
        latex_code += " & ".join(map(str, row))
        latex_code += " \\\\\n"

    latex_code += "\\end{bmatrix}"

    return Math(latex_code)


def complex_matrix_to_string(matrix):
    """
    Transform a matrix of complex numbers to strings truncated to 4 decimals.

    Parameters:
    - matrix (numpy.ndarray): The input matrix of complex numbers.

    Returns:
    numpy.ndarray: Matrix of strings.
    """

    def format_complex_number(x):
        if x.real == 0 and x.imag == 0:
            return "0"
        elif x.real == 0:
            return f"{x.imag:.4f}i"
        elif x.imag == 0:
            return f"{x.real:.4f}"
        else:
            return (
                f"{x.real:.4f} + {x.imag:.4f}i"
                if x.imag >= 0
                else f"{x.real:.4f} - {-x.imag:.4f}I"
            )

    formatted_matrix = np.vectorize(format_complex_number)(matrix)
    return formatted_matrix


def cartesian_to_spherical(coords):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
    - coords (list or numpy array): List or array representing Cartesian coordinates [r, theta, phi].

    Returns:
    - numpy array: Spherical coordinates [r, theta, phi].
    """
    r, theta, phi = coords[0], coords[1], coords[2]
    coords[0] = r * np.sin(theta) * np.cos(phi)
    coords[1] = r * np.sin(theta) * np.sin(phi)
    coords[2] = r * np.cos(theta)
    return coords


from qiskit.visualization.bloch import Bloch
import matplotlib.pyplot as plt


def plot_bloch_multiple_vector(bloch_data, title="Bloch Sphere", font_size=16):
    """
    Plots multiple vectors on a Bloch sphere.

    Parameters:
    - bloch_data (dict): A dictionary where keys are labels for the vectors, and values are 3D vectors
                        representing the points to plot on the Bloch sphere.
    - title (str, optional): The title of the plot. Default is 'Bloch Sphere'.
    - font_size (int, optional): Font size for the annotations on the Bloch sphere. Default is 16.

    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object representing the Bloch sphere plot.
    """

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

    B = Bloch(axes=ax, font_size=font_size)
    B.zlabel = ["z", ""]

    for key, value in bloch_data.items():
        B.add_vectors([value])
        B.add_annotation(value, key)

    B.render(title=title)

    return fig
