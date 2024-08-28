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


def convert_to_hashable(obj):
    """Helper function to convert unhashable types to hashable ones."""
    if isinstance(obj, dict):
        return tuple(sorted((k, convert_to_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(convert_to_hashable(x) for x in obj)
    return obj
