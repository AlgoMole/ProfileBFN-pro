# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common utilities for data pipeline tools."""
import contextlib
import shutil
import tempfile
import time
from typing import Optional, List

from absl import logging


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None, debug: bool = False):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        if not debug:
            shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


@contextlib.contextmanager
def open_files(file_list: List[str], mode: str = "r"):
    """Context manager that closes files on exit."""
    fd_list = [open(f, mode) for f in file_list]
    try:
        yield fd_list
    finally:
        for fd in fd_list:
            fd.close()
