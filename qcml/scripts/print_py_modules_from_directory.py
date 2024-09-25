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

import os
import argparse

def print_file_content(file_path):
    """
    Prints the content of a Python (.py) file.
    
    :param file_path: Path to the Python file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        print(f"--- {file_path} ---")
        print(file.read())
        print("\n")

def find_and_print_py_files(directory):
    """
    Recursively find all Python files in the given directory and print their contents.
    
    :param directory: Directory to scan.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print_file_content(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the content of all Python (.py) files in a directory.")
    parser.add_argument("directory", nargs="?", default=os.getcwd(), help="Directory to scan. Defaults to the current directory.")
    
    args = parser.parse_args()
    
    find_and_print_py_files(args.directory)
