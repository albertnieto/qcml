# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing data generating functions for classification tasks."""

from qic.data.bars_and_stripes import generate_bars_and_stripes
from qic.data.hidden_manifold import generate_hidden_manifold_model
from qic.data.hyperplanes import generate_hyperplanes_parity
from qic.data.linearly_separable import generate_linearly_separable
from qic.data.two_curves import generate_two_curves
from qic.data.load_data import get_dataset
#from qic.data.two_curves import generate_two_curves
#from qic.data.two_curves import generate_two_curves