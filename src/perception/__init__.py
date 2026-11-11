"""Simulated RGB-D perception and dynamic object tracking package (Phases 16.2 & 16.3)."""

import os
import sys

if os.name == "nt":
    _lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.isdir(_lib_bin) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib_bin)
