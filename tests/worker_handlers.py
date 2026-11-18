"""Handlers imported by GPUWorker's child process in tests (needs to be importable)."""

import os
import time


class Echo:
    def echo(self, value):
        return value

    def pid(self):
        return os.getpid()

    def boom(self):
        raise ValueError("bad input")

    def oom(self):
        raise RuntimeError("CUDA failed with error out of memory")

    def die(self):
        os._exit(3)

    def slow(self, seconds):
        time.sleep(seconds)
        return "done"

    def unpicklable(self):
        return lambda: None


class FailsToStart:
    def __init__(self):
        raise RuntimeError("no model file")
