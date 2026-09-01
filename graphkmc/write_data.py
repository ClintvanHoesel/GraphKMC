"""
Created on Wed Mar  9 14:43:07 2022

@author: s164097
"""

import os
import warnings


def write_lines_file(path, lines):
    """Write preformatted text lines to ``path``, warning before overwrite."""
    if os.path.exists(path):
        warnings.warn(f"Overwriting {path}")
    with open(path, "w") as f:
        f.writelines(lines)


def list_of_list_to_lines(lines, delim=","):
    """Convert rows of values to newline-terminated, delimited strings."""
    return [f"{delim.join(map(str, line))}\n" for line in lines]


def write_list_of_lists(
    list_of_list, path, header=[""], units=None, comments=None, delim=","
):
    """Serialize tabular data with optional header, units, and comments."""
    if comments:
        list_of_list = [comments] + list_of_list
    if units:
        list_of_list = [units] + list_of_list
    if header:
        list_of_list = [header] + list_of_list
    write_lines_file(path, list_of_list_to_lines(list_of_list, delim))
