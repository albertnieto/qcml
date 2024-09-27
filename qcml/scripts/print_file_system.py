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
import ast
import argparse
import json

# Try to import termcolor for colored output; if not available, fallback to no color
try:
    from termcolor import colored

    TERM_COLOR_AVAILABLE = True
except ImportError:
    TERM_COLOR_AVAILABLE = False

    def colored(text, color=None):
        return (
            text  # Just return the text without any color if termcolor is not available
        )


# Color palette for different types
TYPE_COLORS = {
    "D": "white",  # Directories
    "M": "light_blue",  # Python modules
    "C": "green",  # Classes
    "F": "light_cyan",  # Functions
}


def extract_functions_and_classes(node):
    """
    Recursively extract classes, functions, and nested functions from a Python module.

    :param node: AST node representing either a class or function.
    :return: Dictionary with classes and functions as keys and their nested elements.
    """
    tree = {}

    if isinstance(node, ast.FunctionDef):
        tree[node.name] = {"_type": "F"}

        for body_item in node.body:
            if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tree[node.name].update(extract_functions_and_classes(body_item))

    elif isinstance(node, ast.ClassDef):
        tree[node.name] = {"_type": "C"}

        for body_item in node.body:
            if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                tree[node.name].update(extract_functions_and_classes(body_item))

    return tree


def extract_structure_from_file(file_path):
    """
    Extracts classes and functions from a Python file in a structured way.

    :param file_path: Path to the Python (.py) file.
    :return: Dictionary representing classes and functions within the file.
    """
    structure = {}
    with open(file_path, "r", encoding="utf-8") as file:
        try:
            tree = ast.parse(file.read(), file_path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    structure.update(extract_functions_and_classes(node))
        except SyntaxError:
            pass  # Skip files that cannot be parsed
    return structure


def build_tree(
    directory,
    only_dirs=False,
    include_functions=False,
    show_hidden=False,
    show_pycache=False,
):
    """
    Build the directory tree structure.

    :param directory: The root directory to scan.
    :param only_dirs: Whether to include only directories.
    :param include_functions: Whether to include classes and functions inside .py files.
    :param show_hidden: Whether to show hidden folders (folders that start with a dot).
    :param show_pycache: Whether to show __pycache__ directories (default is False).
    :return: Dictionary representing the directory tree.
    """
    tree = {}
    entries = sorted(
        os.listdir(directory),
        key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x.lower()),
    )

    if not show_hidden:
        entries = [entry for entry in entries if not entry.startswith(".")]

    if not show_pycache:
        entries = [entry for entry in entries if entry != "__pycache__"]

    entries = (
        [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        if only_dirs
        else entries
    )

    for entry in entries:
        path = os.path.join(directory, entry)
        if os.path.isdir(path):
            tree[entry] = build_tree(
                path, only_dirs, include_functions, show_hidden, show_pycache
            )
            tree[entry]["_type"] = "D"  # Mark as directory
        else:
            if entry.endswith(".py"):
                tree[entry] = {}  # Initialize as dictionary before assigning type
                if include_functions:
                    structure = extract_structure_from_file(path)
                    tree[entry].update(structure)
                tree[entry]["_type"] = "M"  # Mark as Python module
            else:
                tree[entry] = {"_type": None}  # For non-Python files

    return tree


def format_with_type(name, entry_type, show_types=False):
    """
    Format the display name with type indicator and color.

    :param name: Name to format.
    :param entry_type: Type indicator ('D', 'M', 'C', 'F').
    :param show_types: Whether to show type indicators before the name.
    :return: Formatted name with type indicator.
    """
    if show_types:
        name = f"[{entry_type}] {name}"

    if TERM_COLOR_AVAILABLE:
        color = TYPE_COLORS.get(entry_type, None)
        if color:
            name = colored(name, color)
    return name


def print_tree(tree, prefix="", show_types=False, colored_output=True):
    """
    Recursively print the directory tree structure.

    :param tree: Dictionary representing the directory tree.
    :param prefix: Prefix for tree structure formatting.
    :param show_types: Whether to show type indicators.
    :param colored_output: Whether to color the output for better readability.
    """
    entries = list(tree.keys())
    total_entries = len(entries)

    for i, entry in enumerate(entries):
        if entry == "_type":
            continue  # Skip type indicator entries

        connector = "└── " if i == total_entries - 1 else "├── "
        entry_type = (
            tree[entry].get("_type", None) if isinstance(tree[entry], dict) else None
        )

        if not entry_type:
            continue

        formatted_entry = format_with_type(entry, entry_type, show_types=show_types)

        print(prefix + connector + formatted_entry)

        if isinstance(tree[entry], dict):
            extension = "    " if i == total_entries - 1 else "│   "
            print_tree(tree[entry], prefix + extension, show_types, colored_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the directory structure.")
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.getcwd(),
        help="Directory to print. Defaults to the current directory.",
    )
    parser.add_argument(
        "--only-dirs", action="store_true", help="Print only directories."
    )
    parser.add_argument(
        "--json", action="store_true", help="Return the directory tree in JSON format."
    )
    parser.add_argument(
        "--include-functions",
        action="store_true",
        help="Include classes and functions of each Python file in the tree.",
    )
    parser.add_argument(
        "--show-types",
        action="store_true",
        help="Show D (directory), M (module), C (class), F (function) indicators.",
    )
    parser.add_argument(
        "--colored",
        action="store_true",
        default=True,
        help="Colored output for better aesthetics (default: True).",
    )
    parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Show hidden directories (those starting with a dot).",
    )
    parser.add_argument(
        "--show-pycache",
        action="store_true",
        help="Show __pycache__ directories (hidden by default).",
    )

    args = parser.parse_args()

    tree = build_tree(
        args.directory,
        args.only_dirs,
        args.include_functions,
        args.show_hidden,
        args.show_pycache,
    )

    if args.json:
        print(json.dumps(tree, indent=4))
    else:
        print_tree(tree, show_types=args.show_types, colored_output=args.colored)
