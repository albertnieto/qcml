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

def print_tree(directory, only_dirs=False, prefix="", hide_pycache=True, hide_ipynb=True):
    """
    Recursively print a directory tree structure.
    
    :param directory: The root directory to print.
    :param only_dirs: Whether to print only directories.
    :param prefix: Prefix for tree structure formatting.
    :param hide_pycache: Whether to hide __pycache__ directories.
    :param hide_ipynb: Whether to hide .ipynb_checkpoints directories.
    """
    entries = sorted(os.listdir(directory), key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x.lower()))
    
    if hide_pycache:
        entries = [entry for entry in entries if entry != "__pycache__"]
    if hide_ipynb:
        entries = [entry for entry in entries if entry != ".ipynb_checkpoints"]

    # If only directories option is enabled
    entries = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))] if only_dirs else entries
    total_entries = len(entries)

    for i, entry in enumerate(entries):
        path = os.path.join(directory, entry)
        connector = "└── " if i == total_entries - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if i == total_entries - 1 else "│   "
            print_tree(path, only_dirs, prefix + extension, hide_pycache, hide_ipynb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the directory structure.")
    parser.add_argument("directory", nargs="?", default=os.getcwd(), help="Directory to print. Defaults to the current directory.")
    parser.add_argument("--only-dirs", action="store_true", help="Print only directories.")
    parser.add_argument("--hide-pycache", action="store_true", default=True, help="Hide __pycache__ directories (default: True).")
    parser.add_argument("--hide-ipynb", action="store_true", default=True, help="Hide .ipynb_checkpoints directories (default: True).")
    parser.add_argument("--show-pycache", action="store_true", help="Show __pycache__ directories (overrides --hide-pycache).")
    parser.add_argument("--show-ipynb", action="store_true", help="Show .ipynb_checkpoints directories (overrides --hide-ipynb).")
    
    args = parser.parse_args()

    hide_pycache = not args.show_pycache
    hide_ipynb = not args.show_ipynb

    print_tree(args.directory, args.only_dirs, hide_pycache=hide_pycache, hide_ipynb=hide_ipynb)
