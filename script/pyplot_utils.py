import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy.typing as npt
from matplotlib.cm import ScalarMappable
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


def set_axes3d_equal(ax: Axes3D) -> None:
    """Set 3D plot axes to equal scale.

    :param ax: 3D axes.
    :return: None.
    """
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    origin = np.mean(limits, axis=1)
    radius = np.max(np.abs(limits[:, 1] - limits[:, 0])) * 0.5
    min_lim, max_lim = origin - radius, origin + radius
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim3d(min_lim[0], max_lim[0])
    ax.set_ylim3d(min_lim[1], max_lim[1])
    ax.set_zlim3d(min_lim[2], max_lim[2])


def map_colors(p3dc: Poly3DCollection, values: npt.ArrayLike, cmap: str = 'viridis') -> ScalarMappable:
    """Color a tri-mesh by per-face values.

    :param p3dc: `Poly3DCollection`, e.g. returned by `ax.plot_trisurf`.
    :param values: #F scalar values per-face.
    :param cmap: Colormap name.

    :return: `ScalarMappable`, e.g. to make a colorbar.
    """
    norm = Normalize()
    colors = get_cmap(cmap)(norm(values))
    p3dc.set_fc(colors)
    return ScalarMappable(cmap=cmap, norm=norm)
