import os
import torch
from collections import defaultdict
from typing import DefaultDict, List


class CudaEventTimer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.enable_nvtx = os.environ.get("NVTE_ENABLE_NVTX", "0") == "1"
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        if self.enable_nvtx:
            torch.cuda.nvtx.range_push(self.name)
        self.start_event.record()

    def stop(self) -> None:
        self.end_event.record()
        if self.enable_nvtx:
            torch.cuda.nvtx.range_pop()

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


class CudaEventTimerCollection:
    timers: DefaultDict[int, List["CudaEventTimer"]] = defaultdict(list)
    iteration: int = 0

    @staticmethod
    def append(timer: "CudaEventTimer") -> None:
        CudaEventTimerCollection.timers[CudaEventTimerCollection.iteration].append(timer)

    @staticmethod
    def extend(timers: List["CudaEventTimer"]) -> None:
        CudaEventTimerCollection.timers[CudaEventTimerCollection.iteration].extend(timers)

    @staticmethod
    def set_iteration(iteration: int) -> None:
        CudaEventTimerCollection.iteration = iteration

    @staticmethod
    def output() -> None:
        torch.cuda.synchronize()
        print("AK Debug: output timers")
        rank = int(os.environ.get("RANK", -1))
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        path = os.environ.get("NVTE_LOG_PATH", "")
        if path == "":
            raise ValueError("NVTE_LOG_PATH is not set")
        file_path = f"{path}/NVTE_TP_{rank}_{local_rank}.txt"
        with open(os.path.abspath(file_path), "a") as f:
            for iteration, iteration_timers in CudaEventTimerCollection.timers.items():
                for timer in iteration_timers:
                    f.write(f"{iteration} {timer.name} {timer.elapsed_time()}\n")
