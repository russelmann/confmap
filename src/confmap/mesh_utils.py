#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

from __future__ import annotations
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
from numpy import typing as npt
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse.csgraph import dijkstra

from confmap.common import FloatArray, IntArray, pairwise
from confmap.comp_utils import tri_angles


def compute_mesh_edges(faces: IntArray, only_manifold: bool = True) -> Tuple[IntArray, IntArray, List[IntArray]]:
    """Compute mesh edges, face-edge navigation, and boundary loops.

    :param faces: #F by 3 vertex indices of triangular faces.
    :param only_manifold: Allow only 2-manifold meshes.
    :return: #E by 2 edges, #F by 3 face edges, list of boundary loops.
    """
    direct_edges = np.r_[faces[:, 1:], faces[:, (2, 0)], faces[:, :2]]
    edges, edge_index, edge_inverse, edge_counts = np.unique(np.sort(direct_edges), axis=0, return_index=True,
                                                             return_inverse=True, return_counts=True)
    if only_manifold and (edge_counts > 2).any():
        raise ValueError('Faces represent a non-manifold mesh.')
    boundary_edge_index = edge_index[edge_counts == 1]
    face_edges = edge_inverse.reshape(3, -1).T
    boundary_map = {edge[0]: edge[1] for edge in direct_edges[boundary_edge_index, :]}
    boundary_loops, boundary_loop, vertex = [], None, None
    while True:
        if vertex is None:
            if not boundary_map:
                break
            vertex = next(iter(boundary_map))
            boundary_loop = deque([vertex])
        else:
            if (next_vertex := boundary_map.pop(vertex)) == boundary_loop[0]:
                boundary_loop.reverse()
                boundary_loops.append(np.array(boundary_loop))
                vertex = None
            else:
                boundary_loop.append(vertex := next_vertex)
    return edges, face_edges, boundary_loops


def face_face_adjacency(faces: IntArray) -> Tuple[IntArray, IntArray]:
    """Compute face-face connectivity with internal indices.

    :param faces: #F by 3 vertex indices of triangular faces.
    :return: #F by 3 face-face neighbors and #F by 3 internal indices, i.e. local opposing vertex index for each
    neighbor.
    """
    # Edge connectivity.
    direct_edges = np.r_[faces[:, 1:], faces[:, (2, 0)], faces[:, :2]]
    edges, edge_inverse = np.unique(np.sort(direct_edges), axis=0, return_inverse=True)
    face_edges = edge_inverse.reshape(3, -1).T
    face_edge_sides = (np.roll(faces, shift=1, axis=1) < np.roll(faces, shift=2, axis=1)).astype(int)
    edge_faces = -np.ones((len(edges), 2), dtype=int)
    edge_faces[face_edges, face_edge_sides] = np.arange(len(faces))[:, np.newaxis]
    edge_face_inds = -np.ones((len(edges), 2), dtype=int)
    edge_face_inds[face_edges, face_edge_sides] = np.arange(3)[np.newaxis, :]
    # Face connectivity.
    face_face = edge_faces[face_edges, 1 - face_edge_sides]
    face_face_ind = edge_face_inds[face_edges, 1 - face_edge_sides]
    return face_face, face_face_ind


def get_vertex_faces(ff: IntArray, ffi: IntArray, vf: IntArray, vfi: IntArray, vertex_id: int) -> IntArray:
    """Get all faces incident to the given vertex ordered around it. If at boundary, list starts and ends at boundary.
    Requires manifold mesh. No checks performed.

    :param ff: Face-face adjacency.
    :param ffi: Face-face adjacency internal index.
    :param vf: Incident face-per-vertex.
    :param vfi: Incident face-per-vertex internal index.
    :param vertex_id: Vertex ID.
    :return: Face IDs.
    """
    face_id = vf[vertex_id]
    fid, ind = face_id, (vfi[vertex_id] + 1) % 3
    faces_fan = deque()
    is_boundary = True
    while fid != -1:
        faces_fan.append(fid)
        fid, ind = ff[fid, ind], (ffi[fid, ind] + 2) % 3
        if fid == face_id:
            is_boundary = False
            break
    if is_boundary:
        # Not complete loop around vertex.
        fid, ind = face_id, (vfi[vertex_id] + 2) % 3
        faces_fan.popleft()
        while fid != -1:
            faces_fan.appendleft(fid)
            fid, ind = ff[fid, ind], (ffi[fid, ind] + 1) % 3
    return np.array(faces_fan)


def cut_faces(faces: IntArray, start_vertices: List[int], cut_tree: Dict[int, Set[int]]) -> IntArray:
    """Cut triangle mesh along series of vertices. Cutting path must not end at a boundary vertex.

    :param faces: #F by 3 triangle mesh indexing.
    :param start_vertices: Vertex indices to start the cuts.
    :param cut_tree: Vertex indices to cut along (must be a tree going along existing edges, no loops!).
    :return: #F by 3 new triangle mesh indexing with (#VC - 1) extra vertices.
    """
    faces = faces.copy()
    ff, ffi = face_face_adjacency(faces)
    vertex_count = np.max(faces) + 1
    vf = -np.ones(vertex_count, dtype=int)
    vf[faces] = np.arange(len(faces))[:, np.newaxis]
    _, vfi = np.where(faces[vf] == np.arange(len(vf))[:, np.newaxis])

    # vf, vfi = deque(vf), deque(vfi)
    stack = deque(start_vertices)
    while stack:
        if (next_vertex_ids := cut_tree.get(cut_vertex_id := stack.pop())) is None:
            continue  # End of cut.
        next_vertex_ids = next_vertex_ids.copy()
        faces_fan = list(get_vertex_faces(ff, ffi, vf, vfi, cut_vertex_id))
        is_boundary = (-1 in ff[faces_fan[0]])
        if not is_boundary:
            # Cut closed faces fan keeping all vertices.
            for i, fid in enumerate(faces_fan):
                ind = np.argmax(faces[fid] == cut_vertex_id)
                if (vid := faces[fid, (ind + 2) % 3]) in next_vertex_ids:
                    stack.append(vid)
                    ind = (ind + 1) % 3
                    next_fid, next_ffi = ff[fid, ind], ffi[fid, ind]
                    ff[next_fid, next_ffi] = ffi[next_fid, next_ffi] = -1
                    ff[fid, ind] = ffi[fid, ind] = -1
                    next_vertex_ids.remove(vid)
                    faces_fan = faces_fan[i + 1:] + faces_fan[:i + 1]
                    break

        # Perform single-edge cuts.
        for i, fid in enumerate(faces_fan):
            if not next_vertex_ids:
                break
            ind = np.argmax(faces[fid] == cut_vertex_id)
            faces[fid, ind] = vertex_count
            if vid := set(faces[fid]) & next_vertex_ids:
                vid, = vid  # Must be exactly one vertex.
                # vf.append(fid)
                # vfi.append(ind)
                stack.append(vid)
                ind = (ind + 1) % 3
                ff[fid, ind] = ffi[fid, ind] = -1
                next_fid = faces_fan[i + 1]
                ind = np.argmax(faces[next_fid] == cut_vertex_id)
                # vf[cut_vertex_id], vfi[cut_vertex_id] = next_fid, ind
                ind = (ind + 2) % 3
                ff[next_fid, ind] = ffi[next_fid, ind] = -1
                vertex_count += 1
                next_vertex_ids.remove(vid)

    return faces


class TriangleMesh:
    """Class containing mesh data and defining mesh operations for conformal map."""

    @property
    def vertices(self) -> FloatArray:
        return self._vertices

    @property
    def vertex_dim(self) -> int:
        return self._vertices.shape[1]

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def faces(self) -> IntArray:
        return self._faces

    @property
    def edges(self) -> IntArray:
        return self._edges

    @property
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def boundary_vertices(self) -> IntArray:
        return self._boundary_vertices

    @property
    def boundary_edges(self) -> IntArray:
        return np.c_[self.boundary_vertices, np.roll(self.boundary_vertices, shift=-1)]

    @property
    def ff(self) -> IntArray:
        return self._ff

    @property
    def ffi(self) -> IntArray:
        return self._ffi

    @property
    def vf(self) -> IntArray:
        return self._vf

    @property
    def vfi(self) -> IntArray:
        return self._vfi

    @property
    def is_readonly(self) -> bool:
        return self._readonly

    def __init__(self, vertices: npt.ArrayLike, faces: npt.ArrayLike, readonly: bool = False):
        """Create new triangle mesh.

        :param vertices: #V by dim ArrayLike of vertex coordinates.
        :param faces: #F by 3 ArrayLike of vertex indices of triangular faces.
        :param readonly: If True, make all mesh data readonly.
        """
        super(TriangleMesh, self).__init__()

        self._faces = np.array(faces, dtype=int)
        if self._faces.ndim != 2:
            raise ValueError(f'Mesh faces must have array dimension 2, not {self._faces.ndim}.')
        if (faces_dim := self._faces.shape[1]) != 3:
            raise ValueError(f'Mesh faces must be triangles, not {faces_dim}-gons.')
        if vertices is None:
            self._vertices = np.empty((np.max(self._faces) + 1, 0), dtype=float)
        else:
            self._vertices = np.array(vertices, dtype=float)
        if self._vertices.ndim != 2:
            raise ValueError(f'Mesh vertices must have array dimension 2, not {self._vertices.ndim}.')

        self._edges, self._face_edges, boundary_loops = compute_mesh_edges(self._faces)
        if 0 < (hole_count := len(boundary_loops) - 1):
            raise ValueError(f'Mesh must have sphere or disc topology but it has {hole_count} holes.')
        self._boundary_vertices = boundary_loops[0] if boundary_loops else np.empty(0, dtype=int)

        self._ff, self._ffi = face_face_adjacency(self.faces)
        self._vf = -np.ones(self.vertex_count, dtype=int)
        self._vf[faces] = np.arange(len(faces))[:, np.newaxis]
        _, self._vfi = np.where(faces[self._vf] == np.arange(len(self._vf))[:, np.newaxis])

        self._faces.setflags(write=False)
        self._edges.setflags(write=False)
        self._boundary_vertices.setflags(write=False)
        self._ff.setflags(write=False)
        self._ffi.setflags(write=False)
        self._vf.setflags(write=False)
        self._vfi.setflags(write=False)
        self._readonly = False
        if readonly:
            self.set_readonly()

    def set_vertices(self, values: npt.ArrayLike) -> None:
        """Set vertex values.

        :param values: #3 by dim values.
        :return: None.
        """
        if self.is_readonly:
            raise RuntimeError('Mesh is readonly.')
        values = np.array(values, dtype=float)
        if values.ndim != 2:
            raise ValueError(f'Vertex values must have array dimension 2, not: {values.ndim}')
        if values.shape[0] != self.vertex_count:
            raise ValueError(f'Vertex dimension does not match vertex count: {values.shape[0]} != {self.vertex_count}')
        self._vertices = values

    def get_vertex_faces(self, vertex_id: int) -> IntArray:
        """Get all faces incident to the given vertex.

        :param vertex_id: Vertex ID.
        :return: Face IDs.
        """
        return get_vertex_faces(self._ff, self._ffi, self._vf, self._vfi, vertex_id)

    def compute_edge_lengths(self) -> FloatArray:
        """Compute mesh edge lengths.

        :return: #E vector of edge lengths.
        """
        return np.linalg.norm(self._vertices[self._edges[:, 0]] - self._vertices[self._edges[:, 1]], axis=1)

    def compute_angle_defects(self) -> FloatArray:
        """Compute mesh angle defects. For internal vertices equal to discrete Gaussian curvature densities and for
        boundary vertices equal to discrete geodesic curvature densities.

        :return: #V vector of angle defects.
        """
        thetas = np.full(self.vertex_count, 2. * np.pi)
        thetas[self.boundary_vertices] = np.pi
        edge_lengths = self.compute_edge_lengths()
        np.subtract.at(thetas, self.faces, tri_angles(edge_lengths[self._face_edges]))
        return thetas

    def compute_laplacian(self, spmatrix_type: Any = csr_matrix) -> spmatrix:
        """Compute cotan-Laplacian matrix on the mesh.

        :return: The sparse cotan-Laplace matrix.
        """
        if not issubclass(spmatrix_type, spmatrix) or spmatrix_type is spmatrix:
            raise ValueError('spmatrix_type must be strictly a subclass of spmatrix.')
        edge_lengths = self.compute_edge_lengths()
        cotans = 1. / np.tan(tri_angles(edge_lengths[self._face_edges]))
        values = np.zeros(self.edge_count)
        np.add.at(values, self._face_edges, cotans)
        values *= 0.5
        diag = np.zeros(self.vertex_count)
        np.add.at(diag, self.edges[:, 0], values)
        np.add.at(diag, self.edges[:, 1], values)
        data = np.concatenate([-values, diag, -values])
        row_ind = np.concatenate([self.edges[:, 0], np.arange(self.vertex_count), self.edges[:, 1]])
        col_ind = np.concatenate([self.edges[:, 1], np.arange(self.vertex_count), self.edges[:, 0]])
        return spmatrix_type((data, (row_ind, col_ind)), shape=(self.vertex_count,) * 2)

    def find_cuts(self, terminal_vertices: Iterable[int]) -> Tuple[List[int], Dict[int, Set[int]]]:
        """Find cutting trees passing through all terminal vertices.

        :return: List of starting vertices and the cut tree.
        """
        terminal_vertices = list(terminal_vertices)
        edge_lengths = self.compute_edge_lengths()
        adjacency_matrix = csr_matrix((edge_lengths, tuple(self.edges.T)), shape=(self.vertex_count,) * 2)
        start_vertices, cut_tree = set(), defaultdict(set)
        if not (bnd := list(self.boundary_vertices)):
            # No boundary, approximate Steiner tree.
            start_vertices = terminal_vertices[-1:]
            terminal_vertices.pop()
            bnd = list(start_vertices)
        for vertex_id in terminal_vertices:
            _, cut_next, _ = dijkstra(adjacency_matrix, indices=bnd, min_only=True, return_predecessors=True,
                                      directed=False)
            cut_path = deque([curr := vertex_id])
            while 0 <= (curr := cut_next[curr]):
                cut_path.append(curr)
            if cut_path[-1] in self.boundary_vertices:
                start_vertices.add(cut_path[-1])
            for v0, v1 in pairwise(reversed(cut_path)):
                cut_tree[v0].add(v1)
                bnd.append(v1)
        return list(start_vertices), cut_tree

    def cut_faces(self, start_vertices: List[int], cut_tree: Dict[int, Set[int]], copy_values: bool = True,
                  **argv) -> TriangleMesh:
        """Cut faces and return a new mesh.

        :param start_vertices: Start vertices.
        :param cut_tree: Cut tree.
        :param copy_values: If True, vertex values are copied into the new mesh.
        :return: The new mesh.
        """
        new_faces = cut_faces(self.faces, start_vertices, cut_tree)
        new_vertex_count = np.max(new_faces) + 1
        if copy_values:
            new_vertices = np.empty((new_vertex_count, self.vertices.shape[1]))
            new_vertices[new_faces] = self.vertices[self.faces]
        else:
            new_vertices = np.empty((new_vertex_count, 0))
        return TriangleMesh(new_vertices, new_faces, **argv)

    def normalize(self) -> None:
        """Normalize vertices. If spherical, use Mobius normalization.

        :return: None.
        """
        if self.is_readonly:
            raise RuntimeError('Mesh is readonly.')
        if self.vertex_dim != 2:
            raise ValueError('Mesh is not 2D.')
        vertices = self.vertices
        vertices -= np.min(vertices, axis=0)
        vertices /= np.max(np.abs(vertices))

    def copy(self, copy_values: bool = True, **argv) -> TriangleMesh:
        """Copy mesh.

        :param copy_values: If True, vertex values are copied into the new mesh.
        :return: The new mesh.
        """
        return TriangleMesh(self.vertices if copy_values else None, self.faces, **argv)

    def set_readonly(self) -> None:
        """Set mesh readonly.

        :return: None.
        """
        self._readonly = True
        self._vertices.setflags(write=False)
