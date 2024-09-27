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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import re
import numpy as np


class GridSearchAnalysis:
    def __init__(self, path):
        self.path = path
        self.df = None
        self._load_csv()

    def _clean_json(self, json_str):
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r"(\w+):", r'"\1":', json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_str}")
            return {}

    def _load_csv(self):
        self.df = pd.read_csv(self.path)
        self.df["kernel_params"] = self.df["kernel_params"].apply(
            lambda x: self._clean_json(x) if pd.notnull(x) else {}
        )
        self.df["transf_params"] = self.df["transf_params"].apply(
            lambda x: self._clean_json(x) if pd.notnull(x) else {}
        )
        print(f"CSV file loaded from {self.path}")

    def plot_kernel_accuracy_range(
        self,
        experiment_name,
        classifier,
        groups,
        group_names=None,
        figsize=(5, 3),
        title=None,
        legend_title=None,
        palette="husl",
    ):
        if self.df is None:
            print("Dataframe is empty. Please load the CSV file first.")
            return

        df_filtered = self.df.loc[
            (self.df["experiment_name"] == experiment_name)
            & (self.df["classifier"] == classifier)
        ].copy()

        # Ensure that the 'kernel_func' column contains only string values
        df_filtered["kernel_func"] = df_filtered["kernel_func"].astype(str)

        # Now safely apply the .str methods
        df_filtered["kernel_func"] = (
            df_filtered["kernel_func"]
            .str.replace("_kernel", "", regex=False)
            .str.replace("_", " ")
        )

        plt.figure(figsize=figsize)

        palette_colors = sns.color_palette(palette, len(groups))

        xticks = []
        for i, group in enumerate(groups):
            group_df = df_filtered[df_filtered["kernel_func"].isin(group)]

            accuracy_range = (
                group_df.groupby("kernel_func")["accuracy"]
                .agg(["min", "max"])
                .reset_index()
            )
            accuracy_range["mean"] = (accuracy_range["min"] + accuracy_range["max"]) / 2
            accuracy_range["error"] = accuracy_range["max"] - accuracy_range["mean"]

            group_label = (
                group_names[i]
                if group_names and i < len(group_names)
                else f"Grupo {i+1}"
            )
            sns.scatterplot(
                data=accuracy_range,
                x="kernel_func",
                y="mean",
                color=palette_colors[i],
                marker="o",
                s=100,
                label=group_label,
            )

            plt.errorbar(
                x=accuracy_range["kernel_func"],
                y=accuracy_range["mean"],
                yerr=accuracy_range["error"],
                fmt="none",
                c=palette_colors[i],
                capsize=4,
                elinewidth=2,
                capthick=2,
            )

            xticks.extend(list(accuracy_range["kernel_func"]))
            xticks.append("")  # Ensure separation between groups

        # Remove the last empty string to ensure the last tick (separable) is visible
        xticks = xticks[:-1]

        plt.xticks(np.arange(len(xticks)), xticks, rotation=45, ha="right")

        if title:
            plt.title(title)

        plt.xlabel("Función de kernel")
        plt.ylabel("Precisión")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        if legend_title:
            plt.legend(title=legend_title, loc="lower right")
        else:
            plt.legend(loc="lower right")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.show()

    def plot_dataset_evolution(
        self,
        datasets,
        classifiers,
        figsize=(7, 5),
        palette="RdBu",
        title=None,
        x_labels=None,
    ):
        if self.df is None:
            print("Dataframe is empty. Please load the CSV file first.")
            return

        df_filtered = self.df[
            self.df["experiment_name"].isin(datasets)
            & self.df["classifier"].isin(classifiers)
        ].copy()

        plt.figure(figsize=figsize)

        palette_colors = sns.color_palette(palette, len(classifiers))

        for i, classifier in enumerate(classifiers):
            classifier_df = df_filtered[df_filtered["classifier"] == classifier]

            evolution_df = (
                classifier_df.groupby("experiment_name")["accuracy"]
                .agg(["min", "max"])
                .reset_index()
            )
            evolution_df["mean"] = (evolution_df["min"] + evolution_df["max"]) / 2

            sns.lineplot(
                data=evolution_df,
                x="experiment_name",
                y="mean",
                color=palette_colors[i],
                label=classifier,
                linewidth=2,
            )
            plt.fill_between(
                evolution_df["experiment_name"],
                evolution_df["min"],
                evolution_df["max"],
                color=palette_colors[i],
                alpha=0.3,
            )

        plt.xlabel("Dataset")
        plt.ylabel("Accuracy")

        if x_labels:
            plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45, ha="right")
        else:
            plt.xticks(rotation=45, ha="right")

        if title:
            plt.title(title)

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(title="Classifier", loc="lower right")
        plt.tight_layout()
        plt.show()
