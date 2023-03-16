import numpy as np
import matplotlib.pyplot as plt
from vedo import Mesh, colors
import time

import gustaf as gus


def parametrization_function(x):
    return tuple(
        [0.3 - 0.4 * np.maximum(abs(0.5 - x[:, 0]), abs(0.5 - x[:, 1]))]
    )


def parametrization_function_nut(x):
    return tuple([np.array([0.3])])


# Test new microstructure

generator = gus.spline.microstructure.Microstructure()
# outer geometry
generator.deformation_function = gus.Bezier(
    degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
).create.extruded(extrusion_vector=[0, 0, 1])
generator.microtile = gus.spline.microstructure.tiles.NutTile3D()
# how many structures should be inside the cube
time_points=[]
m_range = range(1, 50)
for i in m_range:
    start = time.process_time()
    generator.tiling = [i, i, 5]
    # generator.parametrization_function = parametrization_function_nut
    my_ms = generator.create(contact_length=0.4)
    stop = time.process_time()
    time_points.append((stop-start))
    print(f"{i}: {stop-start}")
    """   generator.show(
        use_saved=True,
        knots=True,
        control_points=False,
        title="3D Nuttile Parametrized Microstructure",
        contact_length=0.4,
        resolutions=2,
    )"""

plt.plot(m_range, time_points)
plt.show()


