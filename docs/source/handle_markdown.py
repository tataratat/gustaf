"""Handles the processing of markdown files.

This script is used to process the markdown files in the project that are also
reused in the documentation. The script will replace the relative links in the
markdown files with relative links that are correct when the documentation is
built.

The paths change since the documentation is built from the docs folder and not
from the root of the project.

Author: Clemens Fricke
"""
import os
import re

# Path to this file.
file_path = os.path.abspath(os.path.dirname(__file__))


def get_markdown_link(line: str) -> str:
    """Get the markdown link from a line.

    Args:
        line (str): Line of text.

    Returns:
        str: Markdown link.
    """
    possible = re.findall(r"\[(.*?)\]\((.*?)\)", line)
    return possible if possible else ""


def get_abs_path_from(path: str, sub_folder: str = "") -> str:
    """Get the absolute path from the projects base directory.

    Args:
        path (str): Relative file path given from the projects base directory.

    Returns:
        str: Absolute path to the given file.
    """
    return os.path.abspath(os.path.join(file_path, "../../", sub_folder, path))


# Folder to save the processed markdown files to.
folder_to_save_to = os.path.abspath(get_abs_path_from("docs/md/"))

# List of markdown files that are used in the documentation.
markdown_files = [
    get_abs_path_from("README.md"),
    get_abs_path_from("CONTRIBUTING.md"),
]

a = None
if __name__ == "__main__":
    # Process all markdown files
    for file in markdown_files:
        # read in the content of the markdown file
        with open(file) as f:
            content = f.read()
        # get all links from the markdown file
        links = get_markdown_link(content)
        # generate a set of all local links
        local_link_set = set()
        for item in links:
            if item[1].startswith(tuple(["http", "#"])):
                continue
            local_link_set.add(item[1])
        # replace all local links with the correct relative links
        for item in local_link_set:
            rel_path = os.path.relpath(get_abs_path_from(item), file_path)
            content = content.replace(item, rel_path)
        # save the processed markdown file into the new md folder
        with open(
            os.path.join(folder_to_save_to, os.path.basename(file)), "w"
        ) as f:
            f.write(content)
