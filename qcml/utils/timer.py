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

import time
from typing import Optional, Callable


class Timer:
    """
    A flexible context manager for timing the execution of code blocks.

    Attributes:
        start_time (float): The start time of the timed block, recorded when entering the block.
        execution_time (float): The elapsed time between entering and exiting the block.
        time_unit (str): The unit of time to use ('seconds', 'milliseconds', 'minutes').
        log_fn (Callable[[str], None]): Optional logging function to log the execution time.
        msg (str): Optional custom message to display or log the execution time.

    Methods:
        __enter__(): Starts the timer when entering the context.
        __exit__(*args): Stops the timer and stores the execution time.
        time(): Returns the recorded execution time after the timer ends.
        show(): Prints the recorded execution time with an optional custom message.
        log(): Logs the recorded execution time using the provided log function.
        reset(): Resets the timer for reuse.
        __str__(): Returns the formatted execution time when printing the object.
        __repr__(): Returns a detailed string with the execution time and attributes.
    """

    def __init__(
        self,
        time_unit: str = "seconds",
        log_fn: Optional[Callable[[str], None]] = None,
        msg: str = "",
    ):
        """
        Initializes the Timer class with optional parameters.

        Args:
            time_unit (str): The unit of time for the execution time ('seconds', 'milliseconds', 'minutes').
            log_fn (Callable[[str], None]): Optional function to log the execution time instead of printing it.
            msg (str): Optional message to prepend to the timing output.
        """
        self.start_time = None
        self.execution_time = None
        self.time_unit = time_unit
        self.log_fn = log_fn
        self.msg = msg

    def __enter__(self):
        """Record the start time when entering the context."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Calculate and store the execution time when exiting the context."""
        end_time = time.time()
        self.execution_time = self._calc_time(end_time)

    def _calc_time(self, end_time: float) -> float:
        """Convert the time difference based on the specified time unit."""
        elapsed_time = end_time - self.start_time
        if self.time_unit == "milliseconds":
            return elapsed_time * 1000
        elif self.time_unit == "minutes":
            return elapsed_time / 60
        return elapsed_time  # Default to seconds

    def time(self) -> Optional[float]:
        """Retrieve the execution time after the block has finished."""
        return self.execution_time

    def show(self):
        """Print the recorded execution time after the timer has finished."""
        if self.execution_time is None:
            raise ValueError("Timer has not finished or hasn't been started.")

        print(self.__str__())

    def log(self):
        """Log the recorded execution time using the provided log function."""
        if self.execution_time is None:
            raise ValueError("Timer has not finished or hasn't been started.")

        if not self.log_fn:
            raise ValueError("No logging function provided.")

        self.log_fn(self.__str__())

    def reset(self):
        """Reset the timer for reuse in a new block."""
        self.start_time = None
        self.execution_time = None

    def __str__(self):
        """Return a formatted string of the execution time."""
        if self.execution_time is None:
            return "Timer has not been started or has not finished."

        return f"{self.msg}Execution time: {self.execution_time:.4f} {self.time_unit}"

    def __repr__(self):
        """Return a detailed string representation of the Timer object."""
        return (
            f"Timer(time_unit={self.time_unit}, "
            f"execution_time={self.execution_time}, msg='{self.msg}')"
        )
