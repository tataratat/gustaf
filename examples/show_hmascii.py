import gustaf as gus


def main():
    try:
        hm_tet = gus.io.hmascii.load(
                "../../samples/volumes/tet/3DPipeCorner90Tet.hmascii"
        )
        hm_hex = gus.io.hmascii.load(
                "../../samples/volumes/hex/3DPipeCorner90Hex.hmascii"
        )

    except BaseException:
        raise RuntimeError(
                'Can`t find mesh in sample files. '
                'Make sure that ``gustaf`` and ``samples`` directory are on '
                'the same level!'
        )

    gus.show.show_vedo(
            ["3DPipeCorner90Tet", hm_tet], ["3DPipeCorner90Hex", hm_hex]
    )


if __name__ == "__main__":
    main()
