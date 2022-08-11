import gustaf as gus
import numpy as np

if __name__ == "__main__":
    v = gus.create.volumes.hexa_block_mesh(
        bounds=[[0, 0, 0], [1, 1, 1]],
        resolutions=[4, 5, 6]
        )

    plots = []

    # show original mesh
    plots.append(["v", v.shrink()])

    # show face groups
    for face_group in v.face_groups:
        plots.append([face_group, v.extract_face_group(face_group).shrink()])

    # create volume groups
    v.volume_groups["odd_elements"] = np.arange(v.volumes.shape[0], step=2)

    # show volume groups
    for volume_group in v.volume_groups:
        plots.append([volume_group,
            v.extract_volume_group(volume_group).shrink()])

    try:
        import vedo
        gus.show.show_vedo(*plots)
    except:
        for item in plots:
            print(f"Showing {item[0]}.")
            item[1].show()

