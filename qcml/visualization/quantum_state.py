import matplotlib.pyplot as plt
import numpy as np
import colorsys
import re
import math
from qiskit.quantum_info import Statevector
from circle_notation import CircleNotation
from ..exceptions import QuantumError

class QuantumState:
    """
    QuantumState class for representing quantum states and visualizing them using Circle Notation.

    Attributes:
    - DEFAULT_CN_OUTER_COLOR (str): Default color for outer circles in Circle Notation.
    - DEFAULT_CN_LEN_OUTER_CIRCLE (float): Default length of outer circles in Circle Notation.
    - DEFAULT_CN_CIRCLES_PER_LINE (int): Default number of circles per line in Circle Notation.
    - DEFAULT_CN_X_DISTANCE_PER_CIRCLE (float): Default x-distance between circles in Circle Notation.
    - DEFAULT_CN_Y_DISTANCE_PER_CIRCLE (float): Default y-distance between circles in Circle Notation.
    - DEFAULT_CN_USE_ZERO_PHASE (bool): Default flag indicating whether to use zero phase in Circle Notation.
    """

    DEFAULT_CN_OUTER_COLOR = "black"
    DEFAULT_CN_LEN_OUTER_CIRCLE = 1
    DEFAULT_CN_CIRCLES_PER_LINE = 4
    DEFAULT_CN_X_DISTANCE_PER_CIRCLE = 2.5
    DEFAULT_CN_Y_DISTANCE_PER_CIRCLE = -2.8
    DEFAULT_CN_USE_ZERO_PHASE = False

    def __init__(self, amplitudes, num_qubits, dimension, state_vector=None):
        """
        Initialize a QuantumState instance.

        Parameters:
        - amplitudes (list): List of amplitudes for the quantum state.
        - num_qubits (int): Number of qubits in the quantum state.
        - dimension (int): Dimension of the quantum state.
        - state_vector (optional): The state vector of the quantum state.
        """

        self.amplitudes = amplitudes
        self.num_qubits = num_qubits
        self.dimension = dimension
        self.state_vector = state_vector

        # Calculate phases
        self.phases = [np.angle(a) for a in amplitudes]
        self.global_phase = np.angle(np.prod(amplitudes))
        self.relative_phase = [phase - self.global_phase for phase in self.phases]
        self.zero_phase = [phase - self.phases[0] for phase in self.phases]

        # Calculate probabilities
        self.probabilities = [abs(a) ** 2 for a in amplitudes]

    def __str__(self):
        """
        Return a string representation of the QuantumState instance.

        Returns:
        - str: String representation of the QuantumState.
        """
        return (
            f"Amplitudes: {self.amplitudes}, \nNum Qubits: {self.num_qubits}, "
            f"\nDimension: {self.dimension}, \nGlobal Phase: {self.global_phase}, "
            f"\nRelative Phase: {self.relative_phase}, \nProbabilities: {self.probabilities}, "
            f"\nPhases: {self.phases}"
        )

    def is_state_vector_valid(self, amplitudes):
        """
        Check if the given amplitudes form a valid quantum state.

        Parameters:
        - amplitudes (list): List of amplitudes.

        Returns:
        - bool: True if valid, False otherwise.
        """
        tolerance = 1e-10
        square_sum = sum(abs(amplitude) ** 2 for amplitude in amplitudes)
        return abs(square_sum - 1) < tolerance

    @classmethod
    def from_complex_numbers(cls, complex_numbers: list) -> "QuantumState":
        """
        Create a QuantumState instance from a list of complex numbers.

        Parameters:
        - complex_numbers (list): List of complex numbers representing the quantum state.

        Returns:
        - QuantumState: The created instance.
        """
        # Check if state vector is valid
        if not cls.is_state_vector_valid(cls, complex_numbers):
            raise QuantumError("Invalid quantum state.")

        num_qubits = int(np.log2(len(complex_numbers)))
        dimension = len(complex_numbers)

        return cls(complex_numbers, num_qubits, dimension)
    
    @classmethod
    def from_string(cls, string: str) -> "QuantumState":
        """
        Create a QuantumState instance from a string representation.

        Parameters:
        - string (str): String representation of the quantum state.

        Returns:
        - QuantumState: The created instance.
        """
        # Extracting the complex numbers from the string
        match = re.match(r"\[?(.*?)\]?$", string, re.DOTALL)

        if not match:
            raise QuantumError("Invalid state vector: invalid string representation.")

        complex_numbers_str = match.group(1)

        # Converting the complex numbers string to a list of complex numbers
        complex_numbers = [complex(val) for val in complex_numbers_str.split(",")]

        # Check if state vector is valid
        if not cls.is_state_vector_valid(cls, complex_numbers):
            raise QuantumError("Invalid quantum state.")

        # Assuming num_qubits and dimension can be derived from the length of the state vector
        num_qubits = int(np.log2(len(complex_numbers)))
        dimension = len(complex_numbers)

        # Check if num_qubits is a power of 2
        if not (num_qubits and not num_qubits & (num_qubits - 1)):
            raise QuantumError(
                "Invalid state vector: Number of qubits must be a power of 2."
            )

        # Creating a QuantumState instance
        return cls(complex_numbers, num_qubits, dimension)

    @classmethod
    def from_statevector(cls, sv: "StateVector") -> "QuantumState":
        """
        Create a QuantumState instance from a StateVector instance.

        Parameters:
        - sv (StateVector): The StateVector instance.

        Returns:
        - QuantumState: The created instance.
        """
        # Check if state vector is valid
        if not cls.is_state_vector_valid(cls, sv.data):
            raise QuantumError("Invalid quantum state.")

        return cls(sv.data, sv.num_qubits, sv.dim, sv)

    def rgb_gradient_from_pi(self, angle: float) -> np.ndarray:
        """
        Convert an angle to an RGB color in the range [0, 2*pi].

        Parameters:
        - angle (float): The angle to convert.

        Returns:
        - np.array: RGB color representation in the range [0, 1].
        """
        # Map the angle to the range [0, 1]
        normalized_angle = (angle % (2 * np.pi)) / (2 * np.pi)

        # Map the angle to the hue value in the HSV color space
        hue = normalized_angle
        saturation = 0.8
        value = 1.0

        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return np.array(rgb)

    def draw_circle_notation(
        self,
        use_zero_phase=DEFAULT_CN_USE_ZERO_PHASE,
        outer_color=DEFAULT_CN_OUTER_COLOR,
        len_outer_circle=DEFAULT_CN_LEN_OUTER_CIRCLE,
        circles_per_line=DEFAULT_CN_CIRCLES_PER_LINE,
        x_distance_per_circle=DEFAULT_CN_X_DISTANCE_PER_CIRCLE,
        y_distance_per_circle=DEFAULT_CN_Y_DISTANCE_PER_CIRCLE,
    ):
        """
        Draw circle notation for the quantum state.

        Parameters:
        - use_zero_phase (bool): Flag indicating whether to use 0 phase in Circle Notation.
        - outer_color (str): Color for outer circles in Circle Notation.
        - len_outer_circle (float): Length of outer circles in Circle Notation.
        - circles_per_line (int): Number of circles per line in Circle Notation.
        - x_distance_per_circle (float): X-distance between circles in Circle Notation.
        - y_distance_per_circle (float): Y-distance between circles in Circle Notation.
        """

        label = [
            f"|{bin(i)[2:].zfill(int(math.sqrt(self.dimension)))}⟩"
            for i in range(2**self.dimension)
        ]

        angle = self.zero_phase if use_zero_phase else self.phases
        
        inner_color = [
            self.rgb_gradient_from_pi(phase)
            for phase in (self.zero_phase if use_zero_phase else self.phases)
        ]
        
        len_inner_circle = [prob for prob in self.probabilities]
        omit_line = [probability == 0 for probability in self.probabilities]

        x_offset = [
            (i % circles_per_line) * x_distance_per_circle
            for i in range(self.dimension)
        ]
        
        y_offset = [
            (i // circles_per_line) * y_distance_per_circle
            for i in range(self.dimension)
        ]

        cn = CircleNotation(
            label=label,
            inner_color=inner_color,
            outer_color=[outer_color] * self.dimension,
            angle=angle,
            x_offset=x_offset,
            y_offset=y_offset,
            omit_line=omit_line,
            len_outer_circle=[len_outer_circle] * self.dimension,
            len_inner_circle=len_inner_circle,
        )
        cn.draw_all()
