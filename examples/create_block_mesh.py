import gustaf as gus
import numpy as np

def main():

    mesh_faces = gus.create.faces.quad_block_mesh(        
        bounds = [[0, 0], [2, 2]],
        resolutions = [2, 3])

    mesh_volumes = gus.create.volumes.hexa_block_mesh(
        bounds = [[0., 0., 0.], [1., 1., 1.]],
        resolutions = [2, 3, 4]
    )

    mesh_faces.show()
    mesh_volumes.show()


if __name__ == "__main__":
    main()