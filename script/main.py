import re
from typing import Union
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from confmap.common import FloatArray
from confmap.comp_utils import conformal_equivalence, quasi_conformal_error
from confmap.confmap import BFF, CETM, ConfMap
from confmap.io_utils import read_obj, write_obj
from confmap.mesh_utils import compute_mesh_edges
from script.pyplot_utils import map_colors, set_axes3d_equal


def plot_cm(ax, cm: ConfMap, rep: str, vkind: str = 'scale', cmap: str = 'jet', **argv):
    if rep not in ('source', 'image'):
        raise ValueError(f'Representation must be either "source" or "image", not "{rep}".')

    if isinstance(cm, CETM):
        prefix = 'CETM, '
    elif isinstance(cm, BFF):
        prefix = 'BFF, '
    else:
        prefix = 'Unknown map, '

    uv, uv_faces = cm.image.vertices, cm.image.faces
    if not cm.is_spherical:
        # uv = orient_uv(uv.copy())
        uv = uv.copy()
        uv += (1. - np.max(uv, axis=0) - np.min(uv, axis=0))[np.newaxis, :] * 0.5  # Fit normalized into [0, 1].

    # Map color underlay.
    if rep == 'image':
        if cm.is_spherical:
            p3dc = ax.plot_trisurf(*uv.T, triangles=cm.faces, linewidth=0, antialiased=True)
            set_axes3d_equal(ax)
        else:
            p3dc = None
            ax.set_aspect('equal')
    else:
        p3dc = ax.plot_trisurf(*cm.vertices.T, triangles=cm.faces, linewidth=0, antialiased=True)
        set_axes3d_equal(ax)

    if vkind in ('scale', 'log-scale'):
        u, _ = conformal_equivalence(cm.vertices, cm.faces, uv[uv_faces])
        if rep == 'image':
            ux = np.zeros(len(uv))
            ux[uv_faces] = u[cm.faces]
        else:
            ux = u
        values = ux if vkind == 'log-scale' else np.exp(ux)
        label = f'{prefix}' + ('scale factors' if vkind == 'scale' else 'log-scale factors')
        if p3dc is None:
            mappable = ax.tripcolor(*uv.T, uv_faces, values, cmap=cmap, shading='gouraud')
        else:
            face_vertex = cm.faces[np.arange(len(cm.faces)), np.abs(values[cm.faces]).argmax(axis=1)]
            mappable = map_colors(p3dc, values[face_vertex], cmap=cmap)
    elif vkind == 'qconf':
        values = quasi_conformal_error(cm.vertices[cm.faces], uv[uv_faces])
        if p3dc is None:
            mappable = ax.tripcolor(*uv.T, uv_faces, facecolors=values)
        else:
            mappable = map_colors(p3dc, values, cmap=cmap)
        label = f'{prefix}quasi-conformal error'
    else:
        raise ValueError('Values do not match uvs or triangles.')
        # plt.colorbar(mappable, ax=ax)  # .set_label(label, rotation=90)
    plt.colorbar(mappable, shrink=1. if rep == 'image' else 0.6)

    # Triangulation.
    if p3dc is None:
        ax.triplot(*uv.T, uv_faces, color='w', lw=0.5, alpha=0.3)

    # Boundary.
    if p3dc is None:
        bnd = compute_mesh_edges(uv_faces)[-1][0]
        bnd = np.append(bnd, bnd[0])
        origs = np.isin(bnd, cm.boundary_vertices)
        splits = np.append(np.where(origs[:-1] != origs[1:])[0] + 1, len(bnd))
        start = 0
        is_cut = ~(origs[:2]).all()
        for till in splits:
            ax.plot(*uv[bnd[start:till + 1]].T, ':k' if is_cut else '-k', lw=1)
            start = till
            is_cut = ~is_cut

    # Cone singularities.
    if p3dc is None:
        cones = np.unique(uv_faces[np.isin(cm.faces, cm.cone_singularities)])  # Find all cones in uv layout.
        ax.plot(*uv[cones].T, 'o', c='m', ms=5.5, mew=1.5, mec='w')

    # Main
    ax.set_title(argv.get('title', label))
    # ax.set_xlabel('u')
    # ax.set_ylabel('v')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def write_mesh(file_name, cm: ConfMap, uv, uv_faces):
    write_obj(f'flat.obj', np.c_[uv, np.zeros(uv.shape[0])], cm.faces)
    if cm.is_spherical:
        write_obj(f'sphere.obj', uv, cm.faces)
    else:
        write_obj(f'{file_name}_uv.obj', cm.vertices, cm.faces, uv, uv_faces)
        write_obj(f'flat.obj', np.c_[uv, np.zeros(uv.shape[0])], uv_faces)


def orient_uv(uv: FloatArray) -> FloatArray:
    """Orient uv vertices so that it is centered and uv[0] -> uv[1] is horizontal."""
    uv -= uv.mean(axis=0)
    vec = uv[1] - uv[0]
    angle = np.arctan2(vec[1], vec[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return uv @ rot


def make_confmap(file_name: str, mode: Union[str, list], method: str) -> ConfMap:
    if method == 'BFF':
        CM = BFF
    elif method == 'CETM':
        CM = CETM
    else:
        raise ValueError(f'Method must be "BFF" or "CETM", not "{method}".')
    vertices, faces = read_obj(f'{file_name}')
    cm = CM(vertices, faces)

    # Specific actions.
    if mode == '':
        pass
    elif isinstance(mode, list):
        cm.add_singularities(mode)
    elif match := re.search(r'^s(\d+)(q?)$', mode):
        cm.add_singularities(int(match[1]), quantize=(np.pi / 2. if match[2] else None))
    elif match := re.search(r'^p(\d+)$', mode):
        vb = np.linspace(0, cm.boundary_vertices.size, num=int(match[1]), dtype=int, endpoint=False)
        cm.setup_polygon(cm.boundary_vertices[vb])
    else:
        raise ValueError(f'Unknown mode "{mode}".')

    if isinstance(cm, CETM):
        if not cm.check_triangle_inequality():
            warn('Triangle inequality failed.')
        if not cm.check_conformal():
            warn('Conformity failed.')

    cm.layout(normalize=True)
    return cm


def demo_compare(file_name: str, mode: Union[str, list]):
    # cm, uv, uv_faces = None, None, None
    fig = plt.figure()
    for i, method in enumerate(('CETM', 'BFF')):
        cm = make_confmap(file_name, mode, method)
        proj = {'projection': '3d'} if cm.is_spherical else {}
        plot_cm(fig.add_subplot(2, 2, (i + 1) * 2 - 1, **proj), cm, 'image', 'log-scale')
        if not cm.is_spherical:
            plot_cm(fig.add_subplot(2, 2, (i + 1) * 2, **proj), cm, 'image', 'qconf')
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # plot_cm(ax, cm, 'source', 'log-scale')
    plt.show()


def demo_tools(file_name: str, method: str, output: str = ''):
    fig = plt.figure()
    modes = ['', 's2', 's2q', 'p4']
    titles = ['Minimum distortion', 'Two singularities', 'Two singularities, quantized', 'Rectangular']
    size = 4
    fig.set_size_inches(2 * size + 0.5, len(modes) * size)
    for row, (mode, title) in enumerate(zip(modes, titles)):
        cm = make_confmap(file_name, mode, method)
        plot_cm(fig.add_subplot(len(modes), 2, row * 2 + 1), cm, 'image', 'log-scale', title=title)
        plot_cm(fig.add_subplot(len(modes), 2, row * 2 + 2), cm, 'image', 'qconf', title=title)
        # write_obj(f'mode_{mode}.obj', cm.vertices, cm.faces, cm.image.vertices, cm.image.faces)
    if output:
        fig.savefig(output, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # demo_compare('../data/spherical.obj', 's2q')
    demo_tools('../data/bumpcap.obj', 'BFF', 'demo.png')
