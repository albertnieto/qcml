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

import numpy as np
import sys
import os
import gc
import time
import argparse
import logging

logging.getLogger().setLevel(logging.INFO)
from importlib import import_module
from datetime import datetime
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from qcml.utils.hyperparam_search import (
    read_data,
    construct_hyperparameter_grid,
    csv_to_dict,
)
from qcml.bench.qml_benchmarking.hyperparameter_settings import hyper_parameter_settings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline


np.random.seed(42)

logging.info("cpu count:" + str(os.cpu_count()))


class ModelScoring:
    def __init__(
        self,
        classifier_name,
        results_path="",
        experiment_name="default_exp",
        hyperparameter_scoring=None,
        hyperparameter_refit="accuracy",
        plot_loss=True,
        n_jobs=-1,
        *args,
        **kwargs,
    ):
        if hyperparameter_scoring is None:
            hyperparameter_scoring = ["accuracy", "roc_auc"]

        self.classifier_name = classifier_name
        self.results_path = results_path
        self.hyperparameter_scoring = hyperparameter_scoring
        self.hyperparameter_refit = hyperparameter_refit
        self.plot_loss = plot_loss
        self.n_jobs = n_jobs

        self._validate_arguments()
        self._prepare_results_directory()
        self.hyperparam_grid = self._create_hyperparam_grid(kwargs)
        self._log_settings()

        self.best_hp_path = None
        self.experiment_name = experiment_name

    def _create_hyperparam_grid(self, kwargs):
        # Add model specific arguments to override the default hyperparameter grid
        hyperparam_grid = construct_hyperparameter_grid(
            hyper_parameter_settings, self.classifier_name
        )

        for hyperparam in hyperparam_grid:
            if hyperparam in kwargs:
                override = kwargs[hyperparam]
                if override is not None:
                    hyperparam_grid[hyperparam] = override

        return hyperparam_grid

    def find_optimal_hyperparameters(self, X, y, experiment_name=None):
        Classifier = self._get_classifier()
        classifier = Classifier()
        classifier_name = Classifier.__name__

        if experiment_name is not None:
            self.experiment_name = experiment_name

        results_filename_stem = f"{classifier_name}_{self.experiment_name}_GridSearchCV"
        results_path = self._prepare_results_directory()
        results_file = os.path.join(results_path, results_filename_stem + ".csv")

        if os.path.isfile(results_file):
            logging.warning(f"Cleaning existing results for {results_file}")

        # Fit once
        a = time.time()
        classifier.fit(X, y)
        b = time.time()

        acc_train = classifier.score(X, y)
        logging.info(
            " ".join(
                [
                    classifier_name,
                    "Exp:",
                    self.experiment_name,
                    "Train acc:",
                    str(acc_train),
                    "Time single run",
                    str(b - a),
                ]
            )
        )

        if hasattr(classifier, "loss_history_"):
            if self.plot_loss:
                plt.plot(classifier.loss_history_)
                plt.xlabel("Iterations")
                plt.ylabel("Loss")
                plt.show()

        if hasattr(classifier, "n_qubits_"):
            logging.info(" ".join(["Num qubits", f"{classifier.n_qubits_}"]))

        gs = GridSearchCV(
            estimator=classifier,
            param_grid=self.hyperparam_grid,
            scoring=self.hyperparameter_scoring,
            refit=self.hyperparameter_refit,
            verbose=3,
        ).fit(X, y)

        logging.info("Best hyperparams")
        logging.info(gs.best_params_)

        df = pd.DataFrame.from_dict(gs.cv_results_)
        df.to_csv(os.path.join(results_path, results_filename_stem + ".csv"))

        best_df = pd.DataFrame(
            list(gs.best_params_.items()), columns=["hyperparameter", "best_value"]
        )

        self.best_hp_path = os.path.join(
            results_path, results_filename_stem + "-best-hyperparameters.csv"
        )

        # Save best hyperparameters to a CSV file
        best_df.to_csv(self.best_hp_path, index=False)

    def score(self, X_train, X_test, y_train, y_test, experiment_name=None):
        Classifier = self._get_classifier()
        classifier = Classifier()
        classifier_name = Classifier.__name__

        if experiment_name != "default_exp":
            self.experiment_name = experiment_name

        results_filename_stem = f"{classifier_name}_{experiment_name}_GridSearchCV"
        self.best_hp_path = os.path.join(
            self.results_path, results_filename_stem + "-best-hyperparameters.csv"
        )
        if self.best_hp_path:
            path_out = os.path.join(
                self.results_path,
                results_filename_stem + "-best-hyperparams-results.csv",
            )

            # Load best hyperparameters
            best_hyperparams = csv_to_dict(self.best_hp_path)

            # Score the model
            results_with_best_hyperparams = {"train_acc": [], "test_acc": []}
            for i in range(5):
                classifier = Classifier(**best_hyperparams, random_state=i)
                classifier.fit(X_train, y_train)

                acc_train = classifier.score(X_train, y_train)
                acc_test = classifier.score(X_test, y_test)
                results_with_best_hyperparams["train_acc"].append(acc_train)
                results_with_best_hyperparams["test_acc"].append(acc_test)

            print("Results with best hyperparams", results_with_best_hyperparams)
            df = pd.DataFrame.from_dict(results_with_best_hyperparams)
            df.to_csv(os.path.join(path_out))

    def _validate_arguments(self):
        if self.classifier_name is None:
            msg = "A classifier from qic.model and dataset path are required"
            raise ValueError(msg)

    def _prepare_results_directory(self):
        results_path = os.path.join(self.results_path, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        return results_path

    def _log_settings(self):
        logging.info(
            "Running hyperparameter search experiment with the following settings\n"
        )
        logging.info("Classifier: " + self.classifier_name)
        logging.info("Hyperparameter scoring: " + " ".join(self.hyperparameter_scoring))
        logging.info("Hyperparameter refit: " + self.hyperparameter_refit)
        logging.info(
            "Hyperparam grid: "
            + " ".join(
                [
                    f"{key}: {self.hyperparam_grid[key]}"
                    for key in self.hyperparam_grid.keys()
                ]
            )
        )

    def _get_classifier(self):
        module = import_module("qic.models")
        Classifier = getattr(module, self.classifier_name)
        return Classifier
