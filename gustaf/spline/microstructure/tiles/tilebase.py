import numpy as np

from gustaf.spline import base


class TileBase(base.GustafBase):
  """
  Base class for tile objects
  """
  def __init__(self):
     """
     Init Values to None
     """
     self._dim = None
     self._evaluation_points = None
     self._parameter_space_dimension = None

  @property
  def parameter_space_dimension(self):
      """Number of parameters per evaluation point."""
      return self._parameter_space_dimension

  @property
  def evaluation_points(self):
      """Positions in the parametrization function to be evaluated when tile
      " "is constructed prior to composition.

      Parameters
      ----------
      None

      Returns
      -------
      evaluation_points : np.ndarray(6,3)
      """
      return self._evaluation_points

  @property
  def dim(self):
      """Returns dimensionality in physical space of the Microtile.

      Parameters
      ----------
      None

      Returns
      -------
      dim : int
      """
      return self._dim
