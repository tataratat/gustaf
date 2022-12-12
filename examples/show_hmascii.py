import gustaf as gus


def main():
    hm_tet = gus.io.hmascii.load("data/hm_tet.hmascii")
    # hm_hex = gus.io.hmascii.load("data/hm_hex.hmascii")
    hm_hex = gus.io.hmascii.load("data/test.py")

    gus.show.show_vedo(["hm_tet", hm_tet], ["hm_hex", hm_hex])


if __name__ == "__main__":
    main()
