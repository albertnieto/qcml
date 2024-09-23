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

# File: qic/bench/checkpoint.py

import os
import json
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    results,
    classifier_name,
    experiment_name,
    batch_idx,
    dataset_name,
    results_path="checkpoints/",
):
    checkpoint_file_name = (
        f"qcml-{experiment_name}-{classifier_name}-{dataset_name}-batch{batch_idx}.json"
    )
    checkpoint_file_path = os.path.join(results_path, checkpoint_file_name)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    checkpoint_data = {"results": results, "batch_idx": batch_idx}

    with open(checkpoint_file_path, "w") as f:
        json.dump(checkpoint_data, f)

    logger.info(f"Checkpoint saved to {checkpoint_file_path}")


def load_checkpoint(
    classifier_name, experiment_name, dataset_name, results_path="checkpoints/"
):
    classifier_name_str = "+".join([clf.__name__ for clf in classifier_name])
    checkpoint_file_name = (
        f"qcml-{experiment_name}-{classifier_name_str}-{dataset_name}.json"
    )
    checkpoint_file_path = os.path.join(results_path, checkpoint_file_name)

    if not os.path.exists(checkpoint_file_path):
        logger.info("No checkpoint found for this classifier, dataset, and experiment.")
        return None

    with open(checkpoint_file_path, "r") as f:
        checkpoint_data = json.load(f)

    logger.info(f"Checkpoint loaded from {checkpoint_file_path}")
    return checkpoint_data


def delete_checkpoints(
    classifier_name, experiment_name, dataset_name, results_path="checkpoints/"
):
    classifier_name_str = "+".join([clf.__name__ for clf in classifier_name])
    checkpoint_file_name = (
        f"qcml-{experiment_name}-{classifier_name_str}-{dataset_name}.json"
    )
    checkpoint_file_path = os.path.join(results_path, checkpoint_file_name)
    if os.path.exists(checkpoint_file_path):
        os.remove(checkpoint_file_path)
        logger.info(
            f"Checkpoint {checkpoint_file_path} deleted after successful completion of the experiment."
        )
    else:
        logger.info(
            f"No checkpoint found to delete for {classifier_name_str} on {dataset_name}."
        )


def get_highest_batch_checkpoint(
    classifier_name, experiment_name, dataset_name, results_path="checkpoints/"
):
    classifier_name_str = "+".join([clf.__name__ for clf in classifier_name])
    checkpoint_file_pattern = (
        f"qcml-{experiment_name}-{classifier_name_str}-{dataset_name}-batch"
    )

    # List all files in the checkpoints directory that match the pattern
    checkpoint_files = [
        f
        for f in os.listdir(results_path)
        if f.startswith(checkpoint_file_pattern) and f.endswith(".json")
    ]

    if not checkpoint_files:
        logger.debug(
            f"No checkpoint found for {classifier_name_str} on {dataset_name} with experiment {experiment_name}."
        )
        return None, None

    # Extract batch numbers and find the highest
    highest_batch = -1
    latest_checkpoint_file = None
    for file in checkpoint_files:
        try:
            batch_str = file.split("-batch")[-1].split(".json")[0]
            batch_idx = int(batch_str)
            if batch_idx > highest_batch:
                highest_batch = batch_idx
                latest_checkpoint_file = file
        except ValueError:
            continue

    if latest_checkpoint_file:
        checkpoint_file_path = os.path.join(results_path, latest_checkpoint_file)
        with open(checkpoint_file_path, "r") as f:
            checkpoint_data = json.load(f)
        logger.debug(
            f"Checkpoint found: {latest_checkpoint_file} with batch {highest_batch}."
        )
        return checkpoint_data, highest_batch

    return None, None
