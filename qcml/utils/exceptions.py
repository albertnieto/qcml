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
