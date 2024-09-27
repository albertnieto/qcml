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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb


def palette_saturated(palette_name="coolwarm", n_colors=2, midpoint=0.2):
    palette = sns.color_palette(palette_name, n_colors=256)
    start_idx = int(midpoint * 256)
    end_idx = int((1 - midpoint) * 256)
    selected_colors = (
        palette[start_idx : start_idx + n_colors // 2]
        + palette[end_idx - n_colors // 2 : end_idx]
    )
    return selected_colors


def adjust_saturation(colors, factor=1.5):
    """Adjust the saturation of a list of colors."""
    colors = np.array(colors)
    hsv_colors = rgb_to_hsv(colors[:, :3])
    hsv_colors[:, 1] = np.clip(hsv_colors[:, 1] * factor, 0, 1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors
