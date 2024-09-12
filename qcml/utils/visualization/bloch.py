import numpy as np
import matplotlib.pyplot as plt
from qutip import Bloch
import pennylane as qml
from pennylane import numpy as pnp

# Initialize a PennyLane device
dev = qml.device("default.qubit", wires=1)

# Function to plot a state on the Bloch sphere
def plot_bloch_vector(vectors, title="", color="b"):
    bloch = Bloch()
    for vec in vectors:
        bloch.add_vectors(vec)
    bloch.vector_color = [color] * len(vectors)  # Set color for all vectors
    bloch.show()
    plt.title(title)
    plt.show()

# Function to compute the Bloch vector from a quantum state
def compute_bloch_vector(state):
    alpha, beta = state[0], state[1]
    X = 2 * np.real(alpha * np.conj(beta))
    Y = 2 * np.imag(alpha * np.conj(beta))
    Z = np.abs(alpha)**2 - np.abs(beta)**2
    return [X, Y, Z]

# Function to plot angle embedding using PennyLane
def plot_angle_embedding(thetas):
    vectors = []

    @qml.qnode(dev)
    def circuit(theta):
        qml.AngleEmbedding(features=[theta], wires=[0], rotation='Y')
        return qml.state()

    for theta in thetas:
        state = circuit(theta)
        bloch_vector = compute_bloch_vector(state)
        vectors.append(bloch_vector)

    plot_bloch_vector(vectors, title="Bloch Sphere: Angle Embedding (R_y(θ))", color="b")

# Angle embedding example values from 1 to 10
theta_angle = [x * np.pi / 10 for x in range(1, 11)]

# Plot the Bloch sphere for the angle embedding
plot_angle_embedding(theta_angle)

# Function to plot amplitude embedding using PennyLane
def plot_amplitude_embedding(pairs):
    vectors = []

    @qml.qnode(dev)
    def circuit(x0, x1):
        qml.AmplitudeEmbedding(features=[x0, x1], wires=[0], normalize=True)
        return qml.state()

    for x0, x1 in pairs:
        state = circuit(x0, x1)
        bloch_vector = compute_bloch_vector(state)
        vectors.append(bloch_vector)

    plot_bloch_vector(vectors, title="Bloch Sphere: Amplitude Embedding", color="r")

# Amplitude embedding example pairs from (1, 2) to (9, 10)
amplitude_pairs = [(x, x + 1) for x in range(1, 10)]

# Plot the Bloch sphere for the amplitude embedding
plot_amplitude_embedding(amplitude_pairs)

# Function to plot IQP embedding using PennyLane
def plot_iqp_embedding(data):
    vectors = []

    @qml.qnode(dev)
    def circuit(x):
        qml.templates.IQPEmbedding(x, wires=[0])
        return qml.state()

    for x in data:
        state = circuit([x])
        bloch_vector = compute_bloch_vector(state)
        vectors.append(bloch_vector)

    plot_bloch_vector(vectors, title="Bloch Sphere: IQP Embedding", color="r")

# Example data for IQP embedding
iqp_data = np.linspace(0, np.pi, 10)  # 10 evenly spaced values between 0 and pi

# Plot the Bloch sphere for the IQP embedding
plot_iqp_embedding(iqp_data)
