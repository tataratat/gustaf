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
import pathlib
import re

# Path to this file.
file_path = os.path.abspath(os.path.dirname(__file__))
repo_root = str(pathlib.Path(__file__).resolve()).split("docs")[0]


def get_markdown_links(line: str) -> str:
    """Get the markdown links from a string.

    Args:
        line (str): Text.

    Returns:
        str: Markdown links.
    """
    possible = re.findall(r"\[(.*?)\]\((.*?)\)", line)
    return possible if possible else ""


def get_github_path_from(link):
    """Substitute the path to the github repository.

    This will expand the link to the github repository. This is used to create
    pages that are independent of the documentation. Like for the long
    description on PyPI. Normally we try to use relative links in the files
    to guarantee that the documentation is also available offline. But for
    the long description on PyPI we need to use absolute links to an online
    repository like the github raw files.

    Args:
        link (str): Relative path to the file.

    Returns:
        str: Https path to the file on github.
    """
    return os.path.abspath(link).replace(
        repo_root, "https://raw.githubusercontent.com/tataratat/gustaf/main/"
    )


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


def process_file(
    file: str, relative_links: bool = True, return_content: bool = False
):
    """Process a markdown file.

    Args:
        file (str): Path to the markdown file.
        relative_links (bool, optional):
            Generate relative links. Defaults to False.
        return_content (bool, optional):
            Return the content instead of saving it to a file.
            Defaults to False.

    Returns:
        str:
            Content of the markdown file.
            Only returned if return_content is True.
    """
    # read in the content of the markdown file
    with open(file) as f:
        content = f.read()
    # get all links from the markdown file
    links = get_markdown_links(content)

    for item in links:
        if item[1].startswith(("http", "#")):  # skip http links and anchors
            content = content.replace(
                f"[{item[0]}]({item[1]})",
                f"<a href='{item[1]}'>{item[0]}</a>",
            )
            continue
        elif not relative_links:  # generate links to github repo
            new_path = get_github_path_from(get_abs_path_from(item[1]))
        else:  # generate relative links
            new_path = os.path.relpath(get_abs_path_from(item[1]), file_path)
        content = content.replace(item[1], new_path)

    if return_content:
        return content

    with open(
        os.path.join(folder_to_save_to, os.path.basename(file)), "w"
    ) as f:
        f.write(content)


if __name__ == "__main__":
    # Process all markdown files
    for file in markdown_files:
        process_file(file)
