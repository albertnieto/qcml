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

logger = logging.getLogger(__name__)


def save_results_to_csv(
    results, classifier_name, experiment_name, results_path, by="accuracy", order="desc"
):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

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
