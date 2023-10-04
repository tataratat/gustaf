"""Helper file for examples. Loads sample files from the tataratat/samples."""
import pathlib

import requests

base_url = "https://raw.githubusercontent.com/tataratat/samples/main/"
local_path = pathlib.Path(__file__).parent / "samples/"


def load_sample_file(filename: str, force_reload: bool = False) -> bool:
    """Loads a sample file from the tataratat/samples repository.

    Args:
        filename (str): Path of the file to load.
        force_reload (bool, optional): Loads the file even if it already
        exists. Defaults to False.

    Returns:
        bool: File could be loaded.
    """
    local_file = local_path / filename
    if local_file.exists() and not force_reload:
        return True
    pathlib.Path.mkdir(local_file.parent, parents=True, exist_ok=True)

    url = base_url + filename
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_file, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Can't load {url}, status code: {response.status_code}.")
        return False
