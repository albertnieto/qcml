class QuantumError(Exception):
    """
    Custom exception class for quantum-related errors.
    """
    def __init__(self, message="Quantum error"):
        """
        Initialize the QuantumError instance.

        Parameters:
        - message (str): Error message. Default is "Quantum error".
        """
        super().__init__(message)

class InvalidGateError(QuantumError):
    """Exception raised for invalid quantum gate operations."""
    pass