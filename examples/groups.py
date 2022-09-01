import gustaf as gus
import numpy as np

if __name__ == "__main__":
    for creation_routine in (gus.create.volumes.hexa_block_mesh,
            gus.create.volumes.tet_block_mesh):
        v = creation_routine(
            bounds=[[0, 0, 0], [1, 1, 1]],
            resolutions=[4, 5, 6]
            )

        # show boundaries
        boundary_plot = gus.show.group_plot(v.extract_all_subelement_groups(),
                shrink=0.98)

        # create volume groups
        v.volume_groups["odd_elements"] = np.arange(v.volumes.shape[0], step=2)
        v.volume_groups["even_elements"] = np.arange(start=1,
                stop=v.volumes.shape[0], step=2)

        # show volume groups
        group_plot = gus.show.group_plot(v.extract_all_element_groups(),
                shrink=0.8)

        try:
            import vedo
            gus.show.show_vedo(boundary_plot, group_plot)
        except:
            for item in boundary_plot + group_plot:
                item.show()

