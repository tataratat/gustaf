"""Example showing the import of a hmascii file.
"""
import pathlib

import load_sample_file

import gustaf as gus


def main():
    base_samples_path = pathlib.Path("samples")
    tet_file_path = pathlib.Path("volumes/tet/3DPipeCorner90Tet.hmascii")
    hex_file_path = pathlib.Path("volumes/hex/3DPipeCorner90Hex.hmascii")

    # download the files from the samples repo if they are not loaded already
    load_sample_file.load_sample_file(tet_file_path)
    load_sample_file.load_sample_file(hex_file_path)

    # check direct hmascii load works
    hm_tet = gus.io.hmascii.load(base_samples_path / tet_file_path)
    # check of load via the default load function works
    hm_hex = gus.io.load(base_samples_path / hex_file_path)

    gus.show.show_vedo(
        ["3DPipeCorner90Tet", hm_tet], ["3DPipeCorner90Hex", hm_hex]
    )


if __name__ == "__main__":
    main()
