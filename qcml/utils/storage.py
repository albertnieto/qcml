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

import pandas as pd
import os
import logging
import re
import fnmatch

logger = logging.getLogger(__name__)


def save_results_to_csv(
    results, classifier_name, experiment_name, results_path, by="accuracy", order="desc"
):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if not results:
        logger.warning("No results to save.")
        return

    csv_file_name = f"{classifier_name}-{experiment_name}_best-hypa.csv"
    csv_file_path = os.path.join(results_path, csv_file_name)

    df_results = pd.DataFrame(results)

    # Determine the sorting order
    ascending = True if order == "asc" else False

    # Sort the DataFrame
    df_results = df_results.sort_values(by=by, ascending=ascending)

    # Save the sorted DataFrame to a CSV file
    df_results.to_csv(csv_file_path, index=False)

    logger.info(
        f"Results saved to {csv_file_path}, file size: {os.path.getsize(csv_file_path)} bytes"
    )


def merge_csv_files(
    path,
    delete_duplicates=True,
    replace_empty_with="NA",
    output_file="default.csv",
    input_file=None,
    delete_merged_files=False,
):
    """
    Merges CSV files in the specified directory based on the input_file parameter into a single CSV file.

    Parameters:
    path (str): Directory containing the CSV files.
    delete_duplicates (bool): Whether to delete duplicate rows. Default is True.
    replace_empty_with (str): Value to replace empty fields with. Default is "NA".
    output_file (str): Name of the output file. Default is "default.csv".
    input_file (str or list or None): Specifies which files to include. Can be a single filename,
                                      a pattern (e.g., "file*"), a regex pattern, or a list of such patterns.
                                      If None, all CSV files in the directory will be processed.
    delete_merged_files (bool): Whether to delete the original files after merging. Default is False.

    Returns:
    None. Saves the merged CSV file in the specified directory as the output file.
    """

    # List to store dataframes
    dataframes = []
    total_rows = 0
    total_duplicates = 0

    # If input_file is None, process all CSV files in the directory
    if input_file is None:
        input_file = ["*.csv"]
    elif isinstance(input_file, str):
        input_file = [input_file]

    # Compile regex patterns for matching files
    compiled_patterns = []
    for pattern in input_file:
        compiled_patterns.append(re.compile(fnmatch.translate(pattern)))

    # List to track files that were processed
    processed_files = []

    # Iterate through all files in the specified directory
    for filename in os.listdir(path):
        # Check if the file matches any of the specified patterns
        if any(pat.match(filename) for pat in compiled_patterns):
            file_path = os.path.join(path, filename)
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            original_rows = len(df)
            total_rows += original_rows

            # Replace empty fields with the specified value
            df.fillna(replace_empty_with, inplace=True)
            # Append the dataframe to the list
            dataframes.append(df)

            logger.debug(f"Processed file: {filename} with {original_rows} rows")
            processed_files.append(file_path)

    # Concatenate all dataframes in the list
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.debug(f"Combined DataFrame created with {len(combined_df)} rows")

    # Drop duplicates if the flag is set to True
    if delete_duplicates:
        combined_df_before = len(combined_df)
        combined_df.drop_duplicates(inplace=True)
        dropped_duplicates = combined_df_before - len(combined_df)
        total_duplicates += dropped_duplicates
        logger.debug(f"Dropped {dropped_duplicates} duplicate rows")

    # Define the output file path
    output_file = os.path.join(path, output_file)

    # Write the combined dataframe to a new CSV file
    combined_df.to_csv(output_file, index=False)

    logger.debug(f"Merged CSV file saved at: {output_file}")
    logger.debug(
        f"Total rows processed: {total_rows}, Total duplicates dropped: {total_duplicates}"
    )

    # Delete original files if the delete_merged_files flag is set to True
    if delete_merged_files:
        for file_path in processed_files:
            os.remove(file_path)
            logger.debug(f"Deleted merged file: {file_path}")
