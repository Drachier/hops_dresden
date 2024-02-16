"""
Module to explicitely define commonly used operators in HOPS.
"""
from typing import Tuple

import numpy as np

def creation_annihilation_operator(dimension: int) -> Tuple[np.ndarray, np.ndarray]
    """
    Returns the creation and annihilation operators for a given dimension.
    """
    creation = np.zeros((dimension, dimension))
    annihilation = np.zeros((dimension, dimension))
    for i in range(dimension - 1):
        creation[i, i + 1] = np.sqrt(i + 1)
        annihilation[i + 1, i] = np.sqrt(i + 1)
    return creation, annihilation

def number_operator(dimension: int) -> np.ndarray:
    """
    Returns the number operator for a given dimension.
    """
    return np.diag(np.arange(dimension))

def pauli_operators() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the Pauli operators.
    """
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    return sigma_x, sigma_y, sigma_z