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

import math
import random
import time
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetCount,
    nvmlDeviceGetName,
    nvmlShutdown,
    nvmlDeviceGetUUID,
    nvmlSystemGetDriverVersion,
)


class GPU:
    def __init__(
        self,
        ID,
        uuid,
        load,
        memoryTotal,
        memoryUsed,
        memoryFree,
        driver,
        gpu_name,
        serial,
        display_mode,
        display_active,
        temp_gpu,
    ):
        self.id = ID
        self.uuid = uuid
        self.load = load
        self.memoryUtil = float(memoryUsed) / float(memoryTotal)
        self.memoryTotal = memoryTotal
        self.memoryUsed = memoryUsed
        self.memoryFree = memoryFree
        self.driver = driver
        self.name = gpu_name
        self.serial = serial
        self.display_mode = display_mode
        self.display_active = display_active
        self.temperature = temp_gpu


def safeFloatCast(strNumber):
    try:
        number = float(strNumber)
    except ValueError:
        number = float("nan")
    return number


def getGPUs():
    nvmlInit()
    try:
        device_count = nvmlDeviceGetCount()
        GPUs = []
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            uuid = nvmlDeviceGetUUID(handle).decode("utf-8")
            name = nvmlDeviceGetName(handle).decode("utf-8")
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            temperature = nvmlDeviceGetTemperature(handle, 0)
            driver = nvmlSystemGetDriverVersion().decode("utf-8")
            serial = "N/A"  # pynvml doesn't provide serial info
            display_mode = "N/A"  # Placeholder
            display_active = "N/A"  # Placeholder

            gpu = GPU(
                ID=i,
                uuid=uuid,
                load=utilization.gpu / 100,
                memoryTotal=mem_info.total / (1024**2),
                memoryUsed=mem_info.used / (1024**2),
                memoryFree=mem_info.free / (1024**2),
                driver=driver,
                gpu_name=name,
                serial=serial,
                display_mode=display_mode,
                display_active=display_active,
                temp_gpu=temperature,
            )
            GPUs.append(gpu)
        return GPUs
    finally:
        nvmlShutdown()


def getAvailable(
    order="first",
    limit=1,
    maxLoad=0.5,
    maxMemory=0.5,
    memoryFree=0,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
):
    GPUs = getGPUs()
    GPUavailability = getAvailability(
        GPUs,
        maxLoad=maxLoad,
        maxMemory=maxMemory,
        memoryFree=memoryFree,
        includeNan=includeNan,
        excludeID=excludeID,
        excludeUUID=excludeUUID,
    )
    availableGPUindex = [
        idx for idx in range(len(GPUavailability)) if GPUavailability[idx] == 1
    ]
    GPUs = [GPUs[g] for g in availableGPUindex]

    if order == "first":
        GPUs.sort(key=lambda x: float("inf") if math.isnan(x.id) else x.id)
    elif order == "last":
        GPUs.sort(
            key=lambda x: float("-inf") if math.isnan(x.id) else x.id, reverse=True
        )
    elif order == "random":
        GPUs = random.sample(GPUs, len(GPUs))
    elif order == "load":
        GPUs.sort(key=lambda x: float("inf") if math.isnan(x.load) else x.load)
    elif order == "memory":
        GPUs.sort(
            key=lambda x: float("inf") if math.isnan(x.memoryUtil) else x.memoryUtil
        )

    GPUs = GPUs[: min(limit, len(GPUs))]
    deviceIds = [gpu.id for gpu in GPUs]
    return deviceIds


def getAvailability(
    GPUs,
    maxLoad=0.5,
    maxMemory=0.5,
    memoryFree=0,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
):
    GPUavailability = [
        (
            1
            if (
                gpu.memoryFree >= memoryFree
                and (gpu.load < maxLoad or (includeNan and math.isnan(gpu.load)))
                and (
                    gpu.memoryUtil < maxMemory
                    or (includeNan and math.isnan(gpu.memoryUtil))
                )
                and (gpu.id not in excludeID)
                and (gpu.uuid not in excludeUUID)
            )
            else 0
        )
        for gpu in GPUs
    ]
    return GPUavailability


def getFirstAvailable(
    order="first",
    maxLoad=0.5,
    maxMemory=0.5,
    attempts=1,
    interval=900,
    verbose=False,
    includeNan=False,
    excludeID=[],
    excludeUUID=[],
):
    for i in range(attempts):
        if verbose:
            print(f"Attempting ({i+1}/{attempts}) to locate available GPU.")
        available = getAvailable(
            order=order,
            limit=1,
            maxLoad=maxLoad,
            maxMemory=maxMemory,
            includeNan=includeNan,
            excludeID=excludeID,
            excludeUUID=excludeUUID,
        )
        if available:
            if verbose:
                print(f"GPU {available} located!")
            break
        if i != attempts - 1:
            time.sleep(interval)

    if not available:
        raise RuntimeError(
            f"Could not find an available GPU after {attempts} attempts with {interval} seconds interval."
        )

    return available


def showUtilization(all=False, attrList=None, useOldCode=False):
    GPUs = getGPUs()
    if all:
        attrList = [
            [
                {"attr": "id", "name": "ID"},
                {"attr": "name", "name": "Name"},
                {"attr": "serial", "name": "Serial"},
                {"attr": "uuid", "name": "UUID"},
            ],
            [
                {
                    "attr": "temperature",
                    "name": "GPU temp.",
                    "suffix": "C",
                    "transform": lambda x: x,
                    "precision": 0,
                },
                {
                    "attr": "load",
                    "name": "GPU util.",
                    "suffix": "%",
                    "transform": lambda x: x * 100,
                    "precision": 0,
                },
                {
                    "attr": "memoryUtil",
                    "name": "Memory util.",
                    "suffix": "%",
                    "transform": lambda x: x * 100,
                    "precision": 0,
                },
            ],
            [
                {
                    "attr": "memoryTotal",
                    "name": "Memory total",
                    "suffix": "MB",
                    "precision": 0,
                },
                {
                    "attr": "memoryUsed",
                    "name": "Memory used",
                    "suffix": "MB",
                    "precision": 0,
                },
                {
                    "attr": "memoryFree",
                    "name": "Memory free",
                    "suffix": "MB",
                    "precision": 0,
                },
            ],
            [
                {"attr": "display_mode", "name": "Display mode"},
                {"attr": "display_active", "name": "Display active"},
            ],
        ]
    else:
        if useOldCode:
            print(" ID  GPU  MEM")
            print("--------------")
            for gpu in GPUs:
                print(f" {gpu.id:2d} {gpu.load*100:.0f}% {gpu.memoryUtil*100:.0f}%")
        elif attrList is None:
            attrList = [
                [
                    {"attr": "id", "name": "ID"},
                    {
                        "attr": "load",
                        "name": "GPU",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                    {
                        "attr": "memoryUtil",
                        "name": "MEM",
                        "suffix": "%",
                        "transform": lambda x: x * 100,
                        "precision": 0,
                    },
                ],
            ]

    if not useOldCode:
        headerString = ""
        GPUstrings = [""] * len(GPUs)
        for attrGroup in attrList:
            for attrDict in attrGroup:
                headerString += f'| {attrDict["name"]} '
                headerWidth = len(attrDict["name"])
                minWidth = len(attrDict["name"])

                attrPrecision = (
                    f'.{attrDict["precision"]}' if "precision" in attrDict else ""
                )
                attrSuffix = str(attrDict["suffix"]) if "suffix" in attrDict else ""
                attrTransform = (
                    attrDict["transform"] if "transform" in attrDict else lambda x: x
                )
                for gpu in GPUs:
                    attr = getattr(gpu, attrDict["attr"])
                    attr = attrTransform(attr)
                    if isinstance(attr, float):
                        attrStr = f"{attr:{attrPrecision}f}"
                    elif isinstance(attr, int):
                        attrStr = f"{attr:d}"
                    elif isinstance(attr, str):
                        attrStr = attr
                    else:
                        raise TypeError(
                            f'Unhandled object type ({type(attr)}) for attribute {attrDict["name"]}'
                        )

                    attrStr += attrSuffix
                    minWidth = max(minWidth, len(attrStr))

                headerString += " " * max(0, minWidth - headerWidth)
                minWidthStr = str(minWidth - len(attrSuffix))

                for gpuIdx, gpu in enumerate(GPUs):
                    attr = getattr(gpu, attrDict["attr"])
                    attr = attrTransform(attr)
                    if isinstance(attr, float):
                        attrStr = f"{attr:{minWidthStr}{attrPrecision}f}"
                    elif isinstance(attr, int):
                        attrStr = f"{attr:{minWidthStr}d}"
                    elif isinstance(attr, str):
                        attrStr = f"{attr:{minWidthStr}s}"
                    else:
                        raise TypeError(
                            f'Unhandled object type ({type(attr)}) for attribute {attrDict["name"]}'
                        )

                    attrStr += attrSuffix
                    GPUstrings[gpuIdx] += f"| {attrStr} "

            headerString += "|"
            for gpuIdx in range(len(GPUs)):
                GPUstrings[gpuIdx] += "|"

        headerSpacingString = "-" * len(headerString)
        print(headerString)
        print(headerSpacingString)
        for GPUstring in GPUstrings:
            print(GPUstring)
