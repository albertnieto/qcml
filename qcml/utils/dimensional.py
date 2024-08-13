import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import umap

def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Returns:
    theta: Polar angle (0 to pi)
    phi: Azimuthal angle (0 to 2pi)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # theta ranges from 0 to pi
    phi = np.arctan2(y, x)  # phi ranges from 0 to 2pi
    if phi < 0:
        phi += 2 * np.pi  # Adjust phi to be in the range [0, 2pi]
    return theta, phi

def normalize_to_bloch_sphere(data_transformed):
    """
    Normalize the transformed 3D data to fit within the Bloch sphere angular ranges.

    Parameters:
    data_transformed (numpy.ndarray): Transformed 3D data array.

    Returns:
    numpy.ndarray: Normalized data array with theta in [0, pi] and phi in [0, 2pi].
    """
    normalized_data = []
    for point in data_transformed:
        x, y, z = point
        theta, phi = cartesian_to_spherical(x, y, z)
        normalized_data.append([theta, phi])
    return np.array(normalized_data)

def pca_transform_to_bloch_sphere(data):
    """
    Transforms high-dimensional data to 3D using PCA and normalizes to fit within the Bloch sphere.

    Parameters:
    data (numpy.ndarray): The input data array of shape (n_samples, n_features).

    Returns:
    numpy.ndarray: The transformed and normalized data array.
    """
    # Handle 1D or 2D data
    if data.shape[1] == 1:
        data = np.hstack((data, np.zeros((data.shape[0], 2))))
    elif data.shape[1] == 2:
        data = np.hstack((data, np.zeros((data.shape[0], 1))))
    
    pca = PCA(n_components=3)
    data_transformed = pca.fit_transform(data)
    
    # Normalize to fit within the Bloch sphere
    norms = np.linalg.norm(data_transformed, axis=1, keepdims=True)
    data_normalized = data_transformed / norms
    
    # Convert to spherical coordinates and normalize to the Bloch sphere's angular ranges
    return normalize_to_bloch_sphere(data_normalized)

def tsne_transform_to_bloch_sphere(data):
    """
    Transforms high-dimensional data to 3D using t-SNE and normalizes to fit within the Bloch sphere.

    Parameters:
    data (numpy.ndarray): The input data array of shape (n_samples, n_features).

    Returns:
    numpy.ndarray: The transformed and normalized data array.
    """
    # Handle 1D or 2D data
    if data.shape[1] == 1:
        data = np.hstack((data, np.zeros((data.shape[0], 2))))
    elif data.shape[1] == 2:
        data = np.hstack((data, np.zeros((data.shape[0], 1))))
    
    tsne = TSNE(n_components=3)
    data_transformed = tsne.fit_transform(data)
    
    # Normalize to fit within the Bloch sphere
    norms = np.linalg.norm(data_transformed, axis=1, keepdims=True)
    data_normalized = data_transformed / norms
    
    # Convert to spherical coordinates and normalize to the Bloch sphere's angular ranges
    return normalize_to_bloch_sphere(data_normalized)

def umap_transform_to_bloch_sphere(data):
    """
    Transforms high-dimensional data to 3D using UMAP and normalizes to fit within the Bloch sphere.

    Parameters:
    data (numpy.ndarray): The input data array of shape (n_samples, n_features).

    Returns:
    numpy.ndarray: The transformed and normalized data array.
    """
    # Handle 1D or 2D data
    if data.shape[1] == 1:
        data = np.hstack((data, np.zeros((data.shape[0], 2))))
    elif data.shape[1] == 2:
        data = np.hstack((data, np.zeros((data.shape[0], 1))))
    
    reducer = umap.UMAP(n_components=3)
    data_transformed = reducer.fit_transform(data)
    
    # Normalize to fit within the Bloch sphere
    norms = np.linalg.norm(data_transformed, axis=1, keepdims=True)
    data_normalized = data_transformed / norms
    
    # Convert to spherical coordinates and normalize to the Bloch sphere's angular ranges
    return normalize_to_bloch_sphere(data_normalized)

def kernel_pca_transform_to_bloch_sphere(data, kernel='rbf', gamma=None):
    """
    Transforms high-dimensional data to 3D using Kernel PCA with a precomputed kernel and normalizes to fit within the Bloch sphere.

    Parameters:
    data (numpy.ndarray): The input data array of shape (n_samples, n_features).
    kernel (str): Kernel type to be used in Kernel PCA (default is 'rbf').
    gamma (float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    Returns:
    numpy.ndarray: The transformed and normalized data array.
    """
    # Handle 1D or 2D data
    if data.shape[1] == 1:
        data = np.hstack((data, np.zeros((data.shape[0], 2))))
    elif data.shape[1] == 2:
        data = np.hstack((data, np.zeros((data.shape[0], 1))))
    
    kpca = KernelPCA(n_components=3, kernel=kernel, gamma=gamma)
    data_transformed = kpca.fit_transform(data)
    
    # Normalize to fit within the Bloch sphere
    norms = np.linalg.norm(data_transformed, axis=1, keepdims=True)
    data_normalized = data_transformed / norms
    
    # Convert to spherical coordinates and normalize to the Bloch sphere's angular ranges
    return normalize_to_bloch_sphere(data_normalized)
