"""NNN: one-line hypothesis. Copy this folder to experiments/NNN_name/ and edit.

    python experiments/NNN_name/run.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common  # noqa: E402

# labels = ...  # array of 199319 'H'/'A'/'D'/'X'
# common.save(__file__, "v1", labels)
# common.score(__file__)
