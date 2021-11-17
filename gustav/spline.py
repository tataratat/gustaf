"""gustav/gustav/spline.py

Spline classes that are inherited from `splinelibpy`, in order to support
visualization and more.
"""

import splinelibpy

class Spline(splinelibpy.Spline):
    def __init__(
        self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
        weights=None,
    ):
        """
        gustav Spline class, derived from splinelibpy.Spline.
        Still an abstact.

        Parameters
        -----------
        degrees: (para_dim,) list-like
        knot_vectors: (para_dim,) list
        control_points: (m, dim) list-like
        weights: (m,) list-like

        Attributes
        -----------
        whatami: str
        para_dim: int
        dim: int
        degrees: np.ndarray
        knot_vectors: list
        control_points: np.ndarray
        knot_vectors_bounds: np.ndarray
        control_points_bounds: np.ndarray
        skip_update: bool

        Returns
        --------
        None
        """
        super().__init__(
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
            weights=weights,
        )

    def show(
            self,
            control_points=True,
            control_point_ids=True,
            control_mesh=False,
            knots=True,
            unique_knot_ids=False,
            resolutions=100,
            quads=True,
            show_queries=True,
            offscreen=False, # <- Implies that it returns plot and objects
            dashed_line=False,
            surface_only=True,
            colorful_elements=False,
            parametric_space=False,
            surface_alpha=1,
            lighting="glossy"
    ):
        """
        Sample, form a mesh, then show.

        Parameters
        -----------
        control_points: bool
          (Optional) Default is True. Show control points.
        control_point_ids: bool
          (Optional) Default is True. Shos index of each control point.
        control_mesh: bool
          (Optional) Default is False. Uses control Mesh instead of collection
          of lines and points. Tends to misbehave in 2D Case, thus, recommended
          to set `False` for 2D Spline Surfaces.
        knots: bool or float
          Show knots. Only for para_dim == 1, this can be float.
          If it is float, it is then used as size for `vedo.Box`.
        unique_knot_ids: bool
          (Optional) Default is False. Shows ids of unique knots,
          only `if not control_points and not knot_ids and parametric_space`.
          `knot_ids` results in too many overlapping ids.
        resolutions: int or list
          (Optional) Default is 100.
        quads: bool
          (Optional) Default is True. If false, triangle.
        offscreen: bool
          (Optional) Default is False. If true, returns `vedo.Plotter` and a
          list of things that should be on the plot.
        surface_only: bool
        colorful_elements: bool
          (Optional) Default is True. Give each element a different color.
          Currently only implemented for lines.
        parametric_space: bool
          (Optional) Default is False. Shows also a view of parametric space.
          Even if `offscreen=True`, it won't return parametric space view.
        suface_alpha: float
          (Optional) Default is 1. Alpha value for surface mesh. Useful to set
          a lower value, if you want a see-through spline.
        lighting: str
          (Optional) Default is glossy. Available options are: [default,
          metallic, plastic, shiny, glossy, ambient, off]. Empty string ("") is
          also allowed.

        Returns
        --------
        plot: `vedo.Plotter`
          (Optional) Only if `offscreen=True`.
        things_to_show: list
          (Optional) Only if `offscreen=True`. List of vedo objects.
        """
        from vedo import show, Points, colors, Line, Box, Plotter

        vedo_colors = [*colors.colors.keys()]
        vedo_colors = [c for c in vedo_colors if not "white" in c]
        things_to_show = []

        if self._para_dim == 1:
            if colorful_elements:
                # TODO: could be cool to sample from a given range
                #   in parametric space -> add `param_range` in `sample()`
                for i in range(int(len(self.unique_knots[0]) - 1)):
                    things_to_show.append(
                        Line(
                            self.evaluate(
                                np.linspace(
                                    self.unique_knots[0][i],
                                    self.unique_knots[0][i+1],
                                    resolutions,
                                ).reshape(-1,1)
                            )
                        ).color(np.random.choice(vedo_colors))
                        .lw(6)
                    )

            else:
                things_to_show.append(
                    self.line(resolutions, dashed_line=False)
                )

            if knots:
                # To utilize automatic sizing, use Points
                ks = Points(
                    self.evaluate(
                        np.asarray(self.unique_knots).T
                    )
                )
                xs = ["x"] * len(self.unique_knots[0])
                things_to_show.append(
                    ks.labels(xs, justify="center", c="green")
                )

        elif self._para_dim == 2:
            if isinstance(resolutions, int):
                resolutions = [resolutions for _ in range(self._para_dim)]

            things_to_show.append(
                self.mesh(
                    resolutions=resolutions,
                    quads=quads,
                    mode="vedo"
                ).color("green").lighting(lighting).alpha(surface_alpha)
            )
            if knots:
                # Only show unique nots -> otherwise, duplicating lines!
                unique_knots = self.unique_knots
                for u in unique_knots[0]:
                    things_to_show.append(
                        self.line(
                            resolution=resolutions[0],
                            raw=False,
                            extract=[0, u],
                            dashed_line=False,
                        )
                    )

                for v in unique_knots[1]:
                    things_to_show.append(
                        self.line(
                            resolution=resolutions[1],
                            raw=False,
                            extract=[1, v],
                            dashed_line=False,
                        )
                    )

            else:
                # Show just edges
                things_to_show.extend(
                    [
                        self.line(
                            resolution=resolutions[0],
                            raw=False,
                            extract=[0, self.knot_vectors[0][0]],
                            dashed_line=False,
                        ),
                        self.line(
                            resolution=resolutions[0],
                            raw=False,
                            extract=[0, self.knot_vectors[0][-1]],
                            dashed_line=False,
                        ),
                        self.line(
                            resolution=resolutions[1],
                            raw=False,
                            extract=[1, self.knot_vectors[1][0]],
                            dashed_line=False,
                        ),
                        self.line(
                            resolution=resolutions[1],
                            raw=False,
                            extract=[1, self.knot_vectors[1][-1]],
                            dashed_line=False,
                        ),
                    ]
                )

        elif self._para_dim == 3:
            if isinstance(resolutions, int):
                resolutions = [resolutions for _ in range(self._para_dim)]
            resolutions = np.asarray(resolutions)

            things_to_show.append(
                self.mesh(
                    resolutions=resolutions,
                    surface_only=surface_only,
                    mode="vedo",
                ).color("green").lighting(lighting).alpha(surface_alpha)
            )

            things_to_show.extend(
                self.lines(
                    resolution=resolutions,
                    outlines=not knots
                )
            )

        if control_points:
            if self._para_dim == 1:
                c_points, c_lines = self.control_mesh(
                    points_and_lines=True,
                    dashed_line=dashed_line,
                )
                things_to_show.extend(
                    [c_points, *c_lines]
                )
                if control_point_ids:
                    things_to_show.append(c_points.labels("id"))

            else:
                if control_mesh:
                    # Mesh
                    c_mesh = self.control_mesh(
                        points_and_lines=False,
                        mode="vedo"
                    )
                    things_to_show.append(
                        c_mesh.c("red").lw(3).wireframe().alpha(.8)
                    )

                    # Points
                    c_points = Points(c_mesh.points(), c="red", r=10, alpha=.8)
                    things_to_show.append(c_points)

                else:
                    c_points, c_lines = self.control_mesh(
                        points_and_lines=True,
                        dashed_line=dashed_line,
                    )
                    things_to_show.extend([c_points, *c_lines])


                if control_point_ids:
                    things_to_show.append(c_points.labels("id"))

        if show_queries and self._fitting_queries is not None:
            things_to_show.append(Points(self._fitting_queries, c="blue", r=15))

        if parametric_space:
            from vedo.addons import Axes

            from .spline_shapes import naive_spline

            ns = naive_spline(self)
            naive_things = ns.show(
                offscreen=True,
                control_points=False,
                surface_alpha=surface_alpha,
                lighting="" if self._para_dim == 2 else lighting
            )
            # Make lines a bit thicker
            for l in naive_things[1:]: l.lw(3)

            # Trick to show begin/end value
            bs = np.asarray(naive_things[0].bounds()).reshape(-1,2).T
            bs_diff_001 = (bs[1] - bs[0]) * 0.001
            lowerb = bs[0] - bs_diff_001 
            upperb = bs[1] + bs_diff_001

            axes_config = dict(
                xtitle="u",
                ytitle="v",
                xrange=[lowerb[0], upperb[0]],
                yrange=[lowerb[1], upperb[1]],
                tipSize=0,
                xMinorTicks=3,
                yMinorTicks=3,
                xyGrid=False,
                yzGrid=False,
            )

            if self._para_dim == 3:
                axes_config.update(ztitle="w")
                axes_config.update(zrange=[lowerb[2], upperb[2]])
                axes_config.update(zMinorTicks=3)
                axes_config.update(zxGrid=False)

            naive_things.append(Axes(naive_things[0], **axes_config))

            if not control_points and unique_knot_ids:
                import itertools

                ks = np.asarray(
                    list(itertools.product(*self.unique_knots))
                )
                para_knots = Points(ks)
                phys_knots = Points(self.evaluate(ks))
                naive_things.append(para_knots.labels("id", c="red"))
                things_to_show.append(phys_knots.labels("id", c="red"))

            if not offscreen:
                plot = Plotter(N=2, sharecam=False)
                plot.show(
                    *naive_things,
                    "Parametric space view",
                    at=0,
                )
                plot.show(
                    things_to_show,
                    "Physical space view",
                    at=1,
                    interactive=True,
                ).close()

            # Always return - don't take it if you don't need it!
            return naive_things, things_to_show

        else:
            if not offscreen:
                show(things_to_show,).close()

            return things_to_show
        

    def mesh(
            self,
            resolutions=100,
            quads=True,
            mode=None,
            surface_only=True
    ):
        """
        Returns spline mesh.

        Warning: Faces of quad meshes are not guaranteed to be coplanar.
          It is okay for visualization using `vedo`. If this is an issue,
          set quads=False, and get triangles.

        Parameters
        -----------
        resolutions: int or list
        quad: bool
        mode: str
          (Optional) options are <"vedo" | "trimesh">.
          If unspecified, regular internal mesh.
        surface_only: bool
          Only for volumes since sampling takes a long long time.

        Returns
        --------
        spline_mesh: `Mesh`
        """
        if isinstance(resolutions, int):
            resolutions = [resolutions for _ in range(self._para_dim)]

        if self._para_dim == 2:
            # Spline Mesh
            physical_points = self.sample(resolutions)
            spline_faces = utils.make_quad_faces(resolutions)

            if not quads:
                spline_faces = utils.diagonalize_quad(spline_faces)

            spline_mesh = Mesh(
                vertices=physical_points,
                faces=spline_faces
            )

        elif self._para_dim == 3:
            if surface_only:
                # Spline to surfaces
                vertices = []
                faces = []
                offset = 0
                for i in range(self._para_dim):
                    extract = i

                    # Get extracting dimension
                    extract_along = [0, 1, 2] 
                    extract_along.pop(extract)

                    # Extract range
                    extract_range = [
                        [
                            min(self.knot_vectors[extract_along[0]]),
                            max(self.knot_vectors[extract_along[0]]),
                        ],
                        [
                            min(self.knot_vectors[extract_along[1]]),
                            max(self.knot_vectors[extract_along[1]]),
                        ],
                    ]

                    extract_list = [
                        min(self.knot_vectors[extract]),
                        max(self.knot_vectors[extract]),
                    ]

                    # surface point queries (spq)
                    spq = np.linspace(
                        extract_range[0][0],
                        extract_range[0][1],
                        resolutions[extract_along[0]],
                    ).reshape(-1, 1)

                    # expand horizontally and init with 1
                    spq = np.hstack((spq, np.ones((len(spq), 1))))

                    spq = np.vstack(
                        np.linspace(
                            spq * [1, extract_range[1][0]],
                            spq * [1, extract_range[1][1]],
                            resolutions[extract_along[1]],
                        )
                    )

                    # expand horizontally and init with 1
                    spq = np.hstack((spq, np.ones((len(spq), 1))))

                    spq = np.vstack(
                        np.linspace(
                            spq * [1, 1, extract_list[0]],
                            spq * [1, 1, extract_list[1]],
                            2
                        )
                    )

                    surface_point_queries = np.ascontiguousarray(
                        spq,
                        dtype=np.double,
                    )
                    surface_point_queries = surface_point_queries[
                        :,
                        np.argsort(
                            [extract_along[0], extract_along[1], extract]
                        )
                    ]
                    vertices.append(
                        self.evaluate(
                            surface_point_queries[
                                :int(surface_point_queries.shape[0] / 2)
                            ]
                        )
                    )

                    if len(faces) != 0:
                        offset = faces[-1].max() + 1

                    tmp_faces = utils.make_quad_faces(
                        [
                            resolutions[extract_along[0]],
                            resolutions[extract_along[1]],
                        ]
                    )

                    faces.append(tmp_faces + int(offset))

                    vertices.append(
                        self.evaluate(
                            surface_point_queries[
                                int(surface_point_queries.shape[0] / 2):
                            ]
                        )
                    )

                    offset = faces[-1].max() + 1

                    faces.append(tmp_faces + int(offset))

                spline_mesh = Mesh(
                    vertices=np.vstack(vertices),
                    faces=np.vstack(faces)
                )

            else:
                # Spline Hexa
                physical_points = self.sample(resolutions)
                spline_elements = utils.make_hexa_elements(resolutions)

                spline_mesh = Mesh(
                    vertices=physical_points,
                    elements=spline_elements
                )

        else:
            logging.debug("Spline - Mesh is only supported for 2D parametric "+\
                "spaces. Skippping.")

        if mode == "vedo":
            spline_mesh = spline_mesh.vedo_mesh
        elif mode == "trimesh":
            spline_mesh = spline_mesh.trimesh_mesh
        else:
            logging.debug("Spline - `mode` is either None or invalid. "+\
                "Returning `gustav.Mesh`.")

        return spline_mesh

    def line(self, resolution, raw=False, extract=None, dashed_line=False):
        """
        Returns line.

        Parameters
        -----------
        resolution: int
        raw: bool
          (Optional) Default is False. Returns vertices and edges.
        extract: list or tuple
          (Optional) Default is None.
          ex) [0, .4] -> [parametric_dim, knot]
          Extracts line from a surface.

        Returns
        --------
        lines: list
          list of vedo.Line
        physical_points: (n, dim) np.ndarray
          (Optional) Only if `raw=True`.
        edges: (m, 2) np.ndarray
          (Optional) Only if `raw=True`.
        """
        if self._para_dim == 1:
            if not raw:
                from vedo import Points, Line, DashedLine

                physical_points = Points(self.sample(resolution))
                if not dashed_line:
                    lines = Line(physical_points, closed=False, c="black", lw=6)

                else:
                    lines = DashedLine(
                        physical_points,
                        closed=False,
                        c="black",
                        lw=6
                    )

                return lines

            else:
                physical_points = self.sample(resolution)
                edges = utils.closed_loop_index_train(physical_points.shape[1])

                return physical_points, edges

        elif self._para_dim == 2:
            if extract is not None:
                # Get non-extracting dimension
                extract_along = [0,1] 
                extract_along.pop(extract[0])

                # Extract range
                extract_range = [
                    min(self.knot_vectors[extract_along[0]]),
                    max(self.knot_vectors[extract_along[0]]),
                ]
                queries = np.zeros((resolution, 2), dtype=np.double)
                queries[:, extract[0]] = extract[1]
                queries[:, extract_along[0]] = np.linspace(
                    extract_range[0],
                    extract_range[1],
                    resolution
                )

                # Extract
                physical_points = self.evaluate(queries)

            else:
                raise ValueError(
                    "To use line() for surface spline, you have to specify "
                    + "a kwarg, `extract`"
                )

            if not raw:
                from vedo import Points, Line, DashedLine

                physical_points = Points(physical_points)
                if not dashed_line:
                    lines = Line(
                        physical_points,
                        closed=False,
                        c="black",
                        lw=2
                    )

                else:
                    lines = DashedLine(
                        physical_points,
                        closed=False,
                        c="black",
                        lw=2
                    )


                return lines

            else:
                edges = utils.open_loop_index_train(
                    physical_points.shape[0]
                )

                return physical_points, edges

    def lines(self, resolution, outlines=False):
        """
        Returns lines. This is meant to be only for volume visualization.
 
        Parameters
        -----------
        resolution: int or list
        outlines: bool

        Returns
        --------
        lines: list
          list of vedo.Line
        """
        if self._para_dim != 3:
            raise ValueError(
                "Sorry, this function (lines_) is only for Solids."
            )

        if not isinstance(resolution, (list, np.ndarray)):
            raise ValueError("For para-dim=3, line extraction needs a "+\
                "list of resolutions")

        from vedo import Points, Line

        # Fill lines
        lines_list = []
        for i in range(self._para_dim):
            extract = [i]

            # Get extracting dimension
            extract_along = [0, 1, 2] 
            extract_along.pop(extract[0])

            # Extract range
            extract_range = [
                [
                    min(self.knot_vectors[extract_along[0]]),
                    max(self.knot_vectors[extract_along[0]]),
                ],
                [
                    min(self.knot_vectors[extract_along[1]]),
                    max(self.knot_vectors[extract_along[1]]),
                ],
            ]

            # Outlines?
            if not outlines:
                unique_knots = self.unique_knots
                last_line_set_queries = list(
                    itertools.product(
                        unique_knots[extract_along[0]],
                        unique_knots[extract_along[1]],

                     )
                )

            else:
                last_line_set_queries = list(
                    itertools.product(
                        extract_range[0],
                        extract_range[1],
                    )
                )

            # Sample lines
            for i, ks in enumerate(last_line_set_queries):
                queries = np.zeros(
                    (resolution[extract[0]], 3),
                    dtype=np.double
                )
                queries[:, extract[0]] = np.linspace(
                    min(self.knot_vectors[extract[0]]),
                    max(self.knot_vectors[extract[0]]),
                    resolution[extract[0]]
                )
                queries[:, extract_along[0]] = ks[0]
                queries[:, extract_along[1]] = ks[1]
                lines_list.append(
                    Line(
                        Points(self.evaluate(queries)),
                        closed=False,
                        c="black",
                        lw=2
                    )
                )

        return lines_list


    def control_mesh(
        self,
        mode=None,
        points_and_lines=False,
        raw=False,
        dashed_line=True,
    ):
        """
        Returns control mesh.

        Parameters
        -----------
        mode: str
          (Optional) options are <"vedo" | "trimesh">.
          If unspecified, regular internal mesh.
          trimesh, will be triangular.
        points_and_lines: bool
        raw: bool

        Returns
        --------
        control_mesh: `Mesh`

        """
        from vedo import Points, Lines, DashedLine

        # Formulate points
        if not raw:
            c_points = Points(self.control_points, c="red", r=10)

        if self._para_dim == 1:
            if not raw:
                c_lines = [DashedLine(c_points, closed=False, c="red", lw=3)]

                return c_points, c_lines

            else:
                return (
                    self.control_points,
                    utils.open_loop_index_train(self.control_points.shape[0]),
                )
            
        elif self._para_dim == 2:
            control_mesh_dims = []
            for i in range(self._para_dim):
                control_mesh_dims.append(
                    len(self.knot_vectors[i]) - self._degrees[i] - 1
                )

            cp_faces = utils.make_quad_faces(control_mesh_dims)
            control_mesh = Mesh(
                vertices=self._control_points,
                faces=cp_faces,
            )

        elif self._para_dim == 3:
            control_mesh_dims = []
            for i in range(self._para_dim):
                control_mesh_dims.append(
                    len(self.knot_vectors[i]) - self._degrees[i] - 1
                )

            cp_elements = utils.make_hexa_elements(control_mesh_dims)
            control_mesh = Mesh(
                vertices=self._control_points,
                elements=cp_elements,
            )

        if mode == "vedo" and not points_and_lines:
            control_mesh = control_mesh.vedo_mesh

        elif mode == "trimesh" and not points_and_lines:
            control_mesh = control_mesh.trimesh_mesh

        elif points_and_lines:
            pass

        else:
            logging.debug("Spline - `mode` is either None or invalid. "+\
                "Returning `gustav.Mesh`.")

        if not points_and_lines:
            return control_mesh

        if not raw:
            c_lines = []
            if dashed_line:
                for ue in control_mesh.unique_edges:
                    c_lines.append(
                        DashedLine( # <- too many objects and could be slow
                            p0=control_mesh.vertices[ue[0]],
                            p1=control_mesh.vertices[ue[1]],
                            c="red",
                            alpha=.8,
                            lw=3,
                        )
                    )

            else:
                c_lines.append(
                    Lines(
                        control_mesh.vertices[control_mesh.unique_edges],
                        c="red",
                        alpha=.8,
                        lw=3,
                    )
                )

        else:
            c_lines = control_mesh.unique_edges
            c_points = control_mesh.vertices

        return c_points, c_lines

