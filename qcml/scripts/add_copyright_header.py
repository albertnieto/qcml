import os

# The copyright header to add
header_to_add = """# Copyright 2024 Albert Nieto

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

# Alternative header that should also be considered valid
alternative_header = """# Copyright 2024 Xanadu Quantum Technologies Inc.
# Modified by Albert Nieto, 2024.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

def has_required_header(content):
    """
    Check if the content of the file already contains the required copyright headers.
    
    :param content: The content of the file as a string.
    :return: True if the file has the required header, otherwise False.
    """
    return (header_to_add in content) or (alternative_header in content)

def add_header_if_missing(file_path):
    """
    Adds the copyright header at the top of the file if it's missing.
    
    :param file_path: Path to the Python file.
    :return: True if the header was added, False otherwise.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if not has_required_header(content):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(header_to_add + '\n' + content)
        return True
    return False

def find_and_add_headers(directory):
    """
    Recursively find all Python files in the given directory and add the copyright header if missing.
    
    :param directory: Directory to scan.
    """
    changed_files = []
    checked_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                checked_files.append(file_path)
                
                if add_header_if_missing(file_path):
                    changed_files.append(file_path)

    # Print out the results
    print(f"\nChecked {len(checked_files)} Python files:")
    for file in checked_files:
        print(f"  - {file}")

    if changed_files:
        print(f"\nModified {len(changed_files)} files:")
        for file in changed_files:
            print(f"  - {file}")
    else:
        print("\nNo files were modified.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add a copyright header to Python files in a directory if missing.")
    parser.add_argument("directory", nargs="?", default=os.getcwd(), help="Directory to scan. Defaults to the current directory.")
    
    args = parser.parse_args()

    find_and_add_headers(args.directory)
