#  MIT License
#  Copyright (c) 2022. Ruslan Guseinov.

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Union
from warnings import warn

import numpy as np
from numpy import typing as npt
from scipy.sparse import csc_matrix, csr_matrix, linalg as sparse_linalg, spdiags

from confmap.common import FloatArray, IntArray
from confmap.comp_utils import lobachevsky, mobius_normalize_layout, quantize_value, quasi_conformal_error, tri_angles
from confmap.geometry import ParametricCurve
from confmap.mesh_utils import TriangleMesh, cut_faces, face_face_adjacency
from confmap.solver import NewtonSolver


class ConfMap(TriangleMesh, ABC):
    """Class defining basic operations for conformal maps.
    """

    @property
    def is_holomorphic(self) -> bool:
        return self._is_holomorphic

    @property
    def is_spherical(self) -> bool:
        return self._is_spherical

    @property
    def cone_singularities(self) -> List[int]:
        return self._cone_singularities.copy()

    @property
    def target_cone_angles(self) -> List[Optional[float]]:
        return self._target_cone_angles.copy()

    @property
    def image(self) -> Optional[TriangleMesh]:
        return self._image

    def __init__(self, vertices: npt.ArrayLike, faces: npt.ArrayLike):
        super(ConfMap, self).__init__(vertices, faces, readonly=True)
        self._image: Optional[TriangleMesh] = None
        self._is_spherical = False
        self._is_holomorphic = True
        self._cone_singularities: List[int] = []
        self._target_cone_angles: List[Optional[float]] = []
        self.reset_cuts()

    def _pre_setup_scale_factors(self, ub: npt.ArrayLike) -> FloatArray:
        ub = np.asarray(ub)
        if ub.ndim != 1:
            raise ValueError('Scale factors must be represented by a 1D array.')
        if ub.size != len(self.boundary_vertices):
            raise ValueError(f'Scale factors must have as many values as boundary vertex count'
                             f' ({ub.size} != {len(self.boundary_vertices)}).')
        return ub

    def reset_cuts(self) -> None:
        self._is_spherical = (self.boundary_vertices.size == 0)  # Topological sphere.
        self._cone_singularities.clear()
        self._target_cone_angles.clear()
        self._post_reset_cuts()

    def _post_reset_cuts(self) -> None:
        pass

    @abstractmethod
    def setup_scale_factors(self, ub: npt.ArrayLike) -> None:
        """Setup conformal map for target boundary log-scale factors optimization.

        :return: None.
        """
        pass

    def setup_minimum_distortion(self) -> None:
        """Setup conformal map for minimum distortion optimization.

        :return: None.
        """
        self.reset_cuts()
        self.setup_scale_factors(np.zeros_like(self.boundary_vertices, dtype=float))

    def _pre_setup_polygon(self, poly_vertices: npt.ArrayLike) -> FloatArray:
        if self.is_spherical:
            raise ValueError('Spherical surface cannot be mapped to a polygon.')
        self._cone_singularities.clear()
        self._target_cone_angles.clear()
        poly_vertices = np.asarray(poly_vertices, dtype=int)
        if poly_vertices.ndim != 1:
            raise ValueError('Polygon must be represented by a 1D array.')
        if poly_vertices.size < 3:
            raise ValueError(f'Polygon must have at least 3 corners (Only {poly_vertices.size} are given).')
        if (inner_vertices := ~np.isin(poly_vertices, self.boundary_vertices)).any():
            raise ValueError(f'Polygon vertices are not at the mesh boundary: {poly_vertices[inner_vertices]}.')
        if len(set(poly_vertices)) != poly_vertices.size:
            raise ValueError(f'Polygon vertices must be unique.')
        return poly_vertices

    @abstractmethod
    def setup_polygon(self, poly_vertices: npt.ArrayLike) -> None:
        """Setup conformal map for polygonal boundary optimization.

        :return: None.
        """
        pass

    def _pre_setup_angles(self, boundary_angles: npt.ArrayLike) -> FloatArray:
        if self.is_spherical:
            raise ValueError('Spherical surface cannot be flattened.')
        boundary_angles = np.asarray(boundary_angles)
        if boundary_angles.ndim != 1:
            raise ValueError('Angles must be represented by a 1D array.')
        if boundary_angles.size != self.boundary_vertices.size:
            raise ValueError(f'Angles do not match boundary length:'
                             f' {boundary_angles.size} != {self.boundary_vertices.size}.')
        return boundary_angles

    @abstractmethod
    def setup_angles(self, boundary_angles: npt.ArrayLike) -> None:
        """Setup conformal map for boundary edge angles optimization. Specify edge angles relative to the real axis for
        edges incoming to boundary vertices.

        :return: None.
        """
        pass

    def _pre_setup_lengths(self, boundary_lengths: npt.ArrayLike) -> FloatArray:
        if self.is_spherical:
            raise ValueError('Spherical surface cannot be flattened.')
        boundary_lengths = np.asarray(boundary_lengths)
        if boundary_lengths.ndim != 1:
            raise ValueError('Edge lengths must be represented by a 1D array.')
        if boundary_lengths.size != self.boundary_vertices.size:
            raise ValueError(f'Edge lengths do not match boundary length:'
                             f' {boundary_lengths.size} != {self.boundary_vertices.size}.')
        return boundary_lengths

    @abstractmethod
    def setup_lengths(self, boundary_lengths: npt.ArrayLike) -> None:
        """Setup conformal map for boundary edge length optimization. Specify edge angles relative to the real axis for
        edges incoming to boundary vertices.

        :return: None.
        """
        pass

    def _pre_add_singularities(self, vertex_ids: npt.ArrayLike,
                               cone_angles: npt.ArrayLike) -> Tuple[IntArray, FloatArray]:
        vertex_ids, cone_angles = np.atleast_1d(np.asarray(vertex_ids)), np.atleast_1d(np.asarray(cone_angles))
        if overlap := set(vertex_ids) & set(self.boundary_vertices):
            raise ValueError(f'Vertices #{list(overlap)} are on the boundary, cannot add cone singularity.')
        if overlap := set(vertex_ids) & set(self._cone_singularities):
            raise ValueError(f'Vertices #{list(overlap)} are already cone singularity.')
        self._cone_singularities.extend(vertex_ids)
        self._target_cone_angles.extend(cone_angles)
        return vertex_ids, cone_angles

    @abstractmethod
    def _add_singularities_by_id(self, vertex_ids: npt.ArrayLike, cone_angles: npt.ArrayLike,
                                 quantize: Optional[float]) -> None:
        """Add cone singularities.

        :param vertex_ids: Vertex ids.
        :param cone_angles: Target cone angles or None to keep them free.
        :param quantize: If not None, round up angle distortion to the nearest value (n * quantize) with n natural.
        :return: None.
        """
        pass

    @abstractmethod
    def _add_singularities_auto(self, count: int, cone_angles: npt.ArrayLike, quantize: Optional[float]) -> None:
        """Pick cone singularities automatically. Select vertices with the largest distortion (CETM strategy).

        :param count: Cones count.
        :param cone_angles: Target cone angles or None to keep them free.
        :param quantize: If not None, round up angle distortion to the nearest value (n * quantize) with n natural.
        :return: None.
        """
        pass

    def add_singularities(self, vertices: npt.ArrayLike, cone_angles: List[Union[float, None]] = None,
                          quantize: float = None) -> None:
        """Add cone singularity.

        :param vertices: Vertex ids or cone count.
        :param cone_angles: Target cone angles or None to keep them free.
        :param quantize: If not None, round up angle distortion to the nearest value (n * quantize) with n natural.
        :return: None.
        """
        vertices = np.asarray(vertices)
        cones_count = len(vertices) if vertices.ndim else int(vertices)
        if cone_angles is None:
            cone_angles = [None] * cones_count
        if cones_count != len(cone_angles):
            raise ValueError(f'Cones count does not match target angles count: {cones_count} != {len(cone_angles)}.')
        if self.is_spherical and len(self._cone_singularities) == 0 and cones_count < 3:
            raise RuntimeError('Spherical map requires at least 3 cone singularities.')
        if vertices.ndim:
            self._add_singularities_by_id(vertices, cone_angles, quantize)
        elif self.is_spherical:
            raise NotImplementedError('Optimal cone singularities for spherical map are not yet implemented.')
            # TODO: Implement optimal cone singularities for spherical map.
        else:
            self._add_singularities_auto(cones_count, cone_angles, quantize)
        if self.is_spherical:
            self._is_spherical = False

    def add_singularity(self, vertex_id: int = None, cone_angle: float = None, quantize: float = None) -> None:
        """Add cone singularity.

        :param vertex_id: Vertex id or None to select vertex with the largest distortion.
        :param cone_angle: Target cone angle or None to keep it free.
        :param quantize: If not None, round up angle distortion to the nearest value (n * quantize) with n natural.
        :return: None.
        """
        self.add_singularities(1 if vertex_id is None else [vertex_id], [cone_angle], quantize)

    @abstractmethod
    def layout(self, normalize: bool = True, start_vertices: List[int] = None,
               cut_tree: Dict[int, Set[int]] = None) -> TriangleMesh:
        """Produce flat layout of the conformal map.

        :param normalize: Normalize map to a unit square.
        :param start_vertices: Vertex indices to start the cuts (must be at boundary) or None for auto.
        :param cut_tree: Vertex indices to cut along (must go along existing edges) or None fot auto.
        :return: Image mesh.
        """
        pass

    @abstractmethod
    def parameterize_uniform(self) -> Tuple[FloatArray, IntArray]:
        """Perform uniformization. Conformal map on a unit circular disk.

        """
        pass

    @abstractmethod
    def parameterize_curve(self, pcurve: ParametricCurve) -> Tuple[FloatArray, IntArray]:
        """Perform parameterization with the given target boundary curve.

        """
        pass


class CETM(ConfMap, NewtonSolver):
    """Class for computation of conformal maps of a triangle mesh using CETM method. Input mesh must have disk topology.
    There are no exhaustive checks, violation will lead to a corrupted outcome.
    """

    @property
    def scale_factors(self) -> FloatArray:
        return self._u

    @property
    def angle_sums(self) -> FloatArray:
        return self._thetas

    @property
    def im_angle_sums(self) -> FloatArray:
        im_thetas = np.zeros(self.vertex_count)
        np.add.at(im_thetas, self.faces, self._im_face_angles)
        return im_thetas

    @property
    def _var(self) -> FloatArray:
        return self._u[self._var_mask]

    @_var.setter
    def _var(self, value: FloatArray) -> None:
        if (self._u[self._var_mask] == value).all():
            return
        self._u[self._var_mask] = value
        self._update_image()

    def __init__(self, vertices: npt.ArrayLike, faces: npt.ArrayLike, polygon: npt.ArrayLike = None):
        """Create CETM. Default is minimum distortion.

        :param vertices: #V by 3 ArrayLike of vertex coordinates.
        :param faces: #F by 3 ArrayLike of vertex indices of triangular faces.
        :param polygon: If not None, boundary of the flat conformal map is polygonal with corners at given vertex ids.
        """
        super(CETM, self).__init__(vertices, faces)
        self._edge_lengths = self.compute_edge_lengths()
        self._edge_lams = 2. * np.log(self._edge_lengths)
        self._u = np.zeros(self.vertex_count)
        self._thetas = np.full(self.vertex_count, 2. * np.pi)
        self._var_mask = np.ones(self.vertex_count, dtype=bool)
        if polygon is not None:
            self.setup_polygon(polygon)
        else:
            self.setup_minimum_distortion()
        if self.is_spherical:
            if polygon is not None:
                raise ValueError('Spherical surface cannot be mapped to a polygon.')
            apex = self.vertex_count - 1
            inactive_faces = self.get_vertex_faces(apex)
            self._var_mask[np.unique(self.faces[inactive_faces])] = False  # Isolate apex.
            self._update_mask()

    def setup_scale_factors(self, ub: npt.ArrayLike) -> None:
        ub = self._pre_setup_scale_factors(ub)
        self._u[:] = 0.
        self._u[self.boundary_vertices] = ub
        self._thetas[:] = 2. * np.pi
        self._var_mask[:] = True
        self._var_mask[self.boundary_vertices] = False
        self._update_mask()

    def setup_polygon(self, poly_vertices: npt.ArrayLike) -> None:
        poly_vertices = self._pre_setup_polygon(poly_vertices)
        self._thetas[:] = 2. * np.pi
        self._thetas[self.boundary_vertices] = np.pi
        self._thetas[poly_vertices] = (1. - 2. / len(poly_vertices)) * np.pi
        self._u[:] = 0.
        self._var_mask[:] = True
        self._var_mask[0] = False
        self._update_mask()

    def setup_angles(self, boundary_angles: npt.ArrayLike) -> None:
        raise NotImplementedError('Direct angle editing is not implemented fro CETM. Use BFF.')

    def setup_lengths(self, boundary_lengths: npt.ArrayLike) -> None:
        raise NotImplementedError('Direct edge length editing is not implemented fro CETM. Use BFF.')

    def _update_mask(self) -> None:
        """Has to be called whenever variable mask is updated."""
        self._var_count = np.sum(self._var_mask)
        self._hess_mask = self._var_mask[self.edges].all(axis=1)
        self._hess_edges = (np.cumsum(self._var_mask) - 1)[self.edges[self._hess_mask]]  # TODO: Improve.
        self._update_image()

    def _update_image(self) -> None:
        """Update image geometric magnitudes dependent on scale factors."""
        self._im_edge_lengths = np.exp(np.mean(self._u[self.edges], axis=1)) * self._edge_lengths
        self._im_face_angles = tri_angles(self._im_edge_lengths[self._face_edges])

    def eval_f(self, var: FloatArray) -> float:
        """Evaluate energy.

        :param var: Values of variable scale factors.
        :return: The energy value.
        """
        self._var = var
        im_edge_lams = self._edge_lams + np.sum(self._u[self.edges], axis=1)
        f = np.sum(im_edge_lams[self._face_edges] * self._im_face_angles) * 0.5
        f += np.sum(lobachevsky(self._im_face_angles.flatten()))
        f -= np.pi * 0.5 * np.sum(self._u[self.faces])  # TODO: Dot product with vertex degree vector.
        f += np.dot(self._u, self._thetas) * 0.5
        return f

    def eval_g(self, var: FloatArray) -> FloatArray:
        """Evaluate gradient of energy.

        :param var: Values of variable scale factors.
        :return: The energy gradient values.
        """
        self._var = var
        g = 0.5 * (self._thetas - self.im_angle_sums)  # Computation has some unused values.
        return g[self._var_mask]

    def eval_H(self, var: FloatArray) -> csr_matrix:
        """Evaluate Hessian of energy.

        :param var: Values of variable scale factors.
        :return: The energy Hessian values.
        """
        self._var = var
        with np.errstate(divide='ignore'):
            cotans = 1. / np.tan(self._im_face_angles)
        cotans[self._im_face_angles == 0.] = 0.  # Correct triangle inequality violations. (Sec. 3.1)
        cotans[self._im_face_angles == np.pi] = 0.
        values = np.zeros(self.edge_count)
        np.add.at(values, self._face_edges, cotans)  # Computation has some unused values.
        diag = np.zeros(self.vertex_count)
        np.add.at(diag, self.edges[:, 0], values)  # Computation has some unused values.
        np.add.at(diag, self.edges[:, 1], values)  # Computation has some unused values.
        diag = diag[self._var_mask]
        data = np.concatenate([diag, -values[self._hess_mask]])
        row_ind = np.concatenate([np.arange(self._var_count), self._hess_edges[:, 0]])
        col_ind = np.concatenate([np.arange(self._var_count), self._hess_edges[:, 1]])
        H = 0.25 * csr_matrix((data, (row_ind, col_ind)), shape=(self._var_count,) * 2)  # Upper triangular.
        # Make symmetric.
        H += H.T
        H.setdiag(H.diagonal() * 0.5)
        return H

    def check_triangle_inequality(self) -> bool:
        """Check violations of triangle inequality.

        :return: True if no violation, otherwise False.
        """
        im_tri_edge_lengths = np.sort(self._im_edge_lengths[self._face_edges], axis=1)  # Sorted!
        return np.all(im_tri_edge_lengths[:, 2] <= np.sum(im_tri_edge_lengths[:, :2], axis=1))

    def check_conformal(self) -> bool:
        """Check violations of conformal map.

        :return: True if no violation, otherwise False.
        """
        im_tri_edge_lengths = self._im_edge_lengths[self._face_edges]
        tri_edge_lengths = self._edge_lengths[self._face_edges]
        ratios = im_tri_edge_lengths / tri_edge_lengths
        prods = np.roll(ratios, shift=-1, axis=1) * np.roll(ratios, shift=1, axis=1)
        tri_u = np.log(tri_edge_lengths / im_tri_edge_lengths * prods)
        u = np.zeros(self.vertex_count)
        u[self.faces] = tri_u
        return np.allclose(tri_u, u[self.faces], atol=1.e-8)

    def _add_singularities_by_id(self, vertex_ids: npt.ArrayLike, cone_angles: npt.ArrayLike,
                                 quantize: Optional[float]) -> None:
        vertex_ids, cone_angles = self._pre_add_singularities(vertex_ids, cone_angles)
        if self.is_spherical:  # Cleanup inactive vertices from the spherical map.
            self._var_mask[:] = True
        self._u[vertex_ids] = 0.
        for vid, angle in zip(vertex_ids, cone_angles):
            self._var_mask[vid] = angle is not None
            if angle is not None:
                self._thetas[vid] = np.pi * 2 - angle
        self._update_mask()
        self.optimize()
        if quantize is not None:
            # TODO: Use TT mesh to find vertex ring?
            for vertex_id in vertex_ids:
                self._var_mask[vertex_id] = True
                im_theta = np.sum(self._im_face_angles[self.faces == vertex_id])  # Inefficient.
                self._thetas[vertex_id] = quantize_value(im_theta, quantize)
            self._update_mask()
            self.optimize()

    def _add_singularities_auto(self, count: int, cone_angles: npt.ArrayLike, quantize: Optional[float]) -> None:
        self.optimize()
        for i in range(count):
            picked = int(np.argmax(np.abs(self._u)))
            self._add_singularities_by_id(picked, cone_angles[i], quantize)

    def layout(self, normalize: bool = True, start_vertices: List[int] = None,
               cut_tree: Dict[int, Set[int]] = None) -> TriangleMesh:
        self.optimize()

        # Perform cuts.
        if self._cone_singularities:
            if start_vertices is None and cut_tree is None:
                start_vertices, cut_tree = self.find_cuts(self._cone_singularities)
            self._image = self.cut_faces(start_vertices, cut_tree, copy_values=False, readonly=False)
        else:
            if start_vertices is not None or cut_tree is not None:
                raise ValueError('Cutting mesh without cone singularities is not allowed.')
            self._image = TriangleMesh.copy(self, copy_values=False, readonly=False)
        faces, ff, ffi = self._image.faces, self._image.ff, self._image.ffi

        uv = np.full((self._image.vertex_count, 2), np.nan)
        visited = np.zeros(self.face_count, dtype=bool)
        start_face = 0
        if self.is_spherical:
            apex = -1
            faces_fan = self.get_vertex_faces(apex)
            visited[faces_fan] = True
            while start_face in faces_fan:
                start_face += 1
        uv[faces[start_face, 1:], :] = 0.
        uv[faces[start_face, 2], 0] = self._im_edge_lengths[self._face_edges[start_face, 0]]  # Layout first edge.
        queue = deque([(start_face, 0)])
        visited[start_face] = True

        while queue:
            face_id, ind = queue.pop()
            next_ind, prev_ind = (ind + 1) % 3, (ind + 2) % 3
            d = uv[faces[face_id, prev_ind]] - uv[faces[face_id, next_ind]]
            d /= np.linalg.norm(d)
            angle = self._im_face_angles[face_id, next_ind]
            length = self._im_edge_lengths[self._face_edges[face_id, prev_ind]]
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            uv[faces[face_id, ind]] = uv[faces[face_id, next_ind]] + (rot @ d) * length
            for i in range(3):
                if (next_face_id := ff[face_id, i]) != -1 and not visited[next_face_id]:
                    queue.appendleft((next_face_id, ffi[face_id, i]))
                    visited[next_face_id] = True

        if self.is_spherical:
            uv = mobius_normalize_layout(uv)
        self._image.set_vertices(uv)
        if normalize and not self.is_spherical:
            self._image.normalize()
        self._image.set_readonly()
        return self._image

    def parameterize_uniform(self) -> Tuple[FloatArray, IntArray]:
        raise NotImplementedError('Uniformization is not implemented for CETM. Use BFF.')

    def parameterize_curve(self, pcurve: ParametricCurve) -> Tuple[FloatArray, IntArray]:
        raise NotImplementedError('Target boundary curve is not implemented for CETM. Use BFF.')


class BFF(ConfMap):
    """Class for computation of conformal maps of a triangle mesh using CETM method. Input mesh must have disk topology.
    There are no exhaustive checks, violation will lead to a corrupted outcome.
    """

    def __init__(self, vertices: npt.ArrayLike, faces: npt.ArrayLike, polygon: npt.ArrayLike = None):
        """Create BFF conformal map. Default is minimum distortion.

        :param vertices: #V by 3 ArrayLike of vertex coordinates.
        :param faces: #F by 3 ArrayLike of vertex indices of triangular faces.
        :param polygon: If not None, boundary of the flat conformal map is polygonal with corners at given vertex ids.
        """
        super(BFF, self).__init__(vertices, faces)
        edge_points = self.vertices[self.boundary_edges]
        self._boundary_edge_lengths = np.linalg.norm(edge_points[:, 1, :] - edge_points[:, 0, :], axis=1)
        self._angle_defects = self.compute_angle_defects()
        self._A = self.compute_laplacian(spmatrix_type=csc_matrix)
        self._A.setdiag(self._A.diagonal() + 1.e-8)  # For solver stability.
        bb = np.zeros(self.vertex_count, dtype=bool)
        bb[self.boundary_vertices] = True
        self._inner_vertices = np.where(~bb)[0]
        AII = self._A[self._inner_vertices[:, np.newaxis], self._inner_vertices]
        self._AIB = self._A[self._inner_vertices[:, np.newaxis], self.boundary_vertices]
        self._ABB = self._A[self.boundary_vertices[:, np.newaxis], self.boundary_vertices]

        # This can be improved using more advanced linear solvers.
        self._solveA = sparse_linalg.factorized(self._A)
        self._solveAII = sparse_linalg.factorized(AII)

        self._vmap_cut, self._bff_cut = None, None
        self._ub, self._im_k = None, None
        if polygon is not None:
            self.setup_polygon(polygon)
        else:
            self.setup_minimum_distortion()

    def _set_dirichlet(self, ub: FloatArray) -> None:
        """Set target log-scale factors along the boundary. Implements DirichletToNeumann.

        :param ub: #B boundary log-scale factors.
        :return: None
        """
        self._ub = ub
        if self._cone_singularities:
            thetas = self._compute_cone_angles()
            ui = -self._solveAII((self._angle_defects - thetas)[self._inner_vertices] + self._AIB @ self._ub)
            u = np.zeros(self.vertex_count)
            u[self.boundary_vertices] = self._ub
            u[self._inner_vertices] = ui
            u_cut = u[self._vmap_cut]
            self._bff_cut._ub = u_cut[self._bff_cut.boundary_vertices]
            h = -self._bff_cut._AIB.T @ u_cut[self._bff_cut._inner_vertices] - self._bff_cut._ABB @ self._bff_cut._ub
            self._bff_cut._im_k = self._bff_cut._angle_defects[self._bff_cut.boundary_vertices] - h
            self._im_k = None  # Becomes meaningless.
        else:
            ui = -self._solveAII(self._angle_defects[self._inner_vertices] + self._AIB @ self._ub)
            h = -self._AIB.T @ ui - self._ABB @ self._ub
            self._im_k = self._angle_defects[self.boundary_vertices] - h

    def _set_neumann(self, im_k: FloatArray) -> None:
        """Set target exterior angles along the boundary. Implements NeumannToDirichlet.

        :param im_k: #B image exterior angles.
        :return: None
        """
        self._im_k = im_k
        om = self._angle_defects.copy()
        om[self.boundary_vertices] -= im_k
        self._ub = self._solveA(-om)[self.boundary_vertices]
        self._ub -= self._ub.mean()

    def _best_fit_curve(self) -> FloatArray:
        """Fit boundary vertices given target boundary properties. Implements BestFitCurve.

        :return: #B by 2 uv-coordinates of boundary curve.
        """
        bff = self._bff_cut if self._cone_singularities else self
        phi = np.cumsum(-bff._im_k)
        tangents = np.c_[np.cos(phi), np.sin(phi)]
        im_elen = np.exp((bff._ub + np.roll(bff._ub, shift=-1)) * 0.5) * bff._boundary_edge_lengths
        boundary_vertex_masses = (bff._boundary_edge_lengths + np.roll(bff._boundary_edge_lengths, shift=1)) * 0.5
        if self._cone_singularities:
            # Matching edges share degrees of freedom (exactly equal edge lengths and tangents).
            edges = self._vmap_cut[self._bff_cut.boundary_edges]
            _, ind, inv = np.unique(np.sort(edges, axis=1), axis=0, return_index=True, return_inverse=True)
            nb = len(ind)
            tangents_uni = np.zeros((nb, 2))
            np.add.at(tangents_uni, inv, tangents)
            im_elen = im_elen[ind]
            boundary_vertex_masses = boundary_vertex_masses[ind]
        else:
            nb, tangents_uni, inv = len(self.boundary_vertices), tangents, None
        N1 = spdiags(boundary_vertex_masses, 0, nb, nb)  # Inverse mass matrix.
        im_elen -= N1 @ tangents_uni @ np.linalg.inv(tangents_uni.T @ N1 @ tangents_uni) @ tangents_uni.T @ im_elen
        if (im_elen <= 0.).any():
            warn('Some image boundary edge lengths are non-positive.')
        if self._cone_singularities:
            im_elen = im_elen[inv]
        return np.roll(np.cumsum(im_elen[:, np.newaxis] * tangents, axis=0), shift=1, axis=0)

    def _extend_curve(self, uvb: FloatArray) -> FloatArray:
        """Reconstruct the complete conformal map give the boundary vertices. Implements ExtendCurve.

        :param uvb: #B by 2 uv-coordinates of boundary curve.
        :return: None
        """
        if self._cone_singularities:
            raise RuntimeError('Extend curve should not be called with cone singularities.')
        h = np.zeros(self.vertex_count)
        h[self.boundary_vertices] = (np.roll(uvb[:, 0], shift=1) - np.roll(uvb[:, 0], shift=-1)) * 0.5
        uv = np.zeros((self.vertex_count, 2))
        uv[self.boundary_vertices, 0] = uvb[:, 0]
        uv[self._inner_vertices, 0] = self._solveAII(-self._AIB @ uvb[:, 0])
        if self.is_holomorphic:
            # Holomorphic extension.
            uv[:, 1] = self._solveA(-h)
        else:
            # Harmonic extension.
            uv[self.boundary_vertices, 1] = uvb[:, 1]
            uv[self._inner_vertices, 1] = self._solveAII(-self._AIB @ uvb[:, 1])
        return uv

    def _post_reset_cuts(self) -> None:
        self._bff_cut, self._vmap_cut = None, None

    def setup_scale_factors(self, ub: npt.ArrayLike) -> None:
        ub = self._pre_setup_scale_factors(ub)
        self._set_dirichlet(ub)
        self._is_holomorphic = True

    def setup_polygon(self, poly_vertices: npt.ArrayLike) -> None:
        poly_vertices = self._pre_setup_polygon(poly_vertices)
        self._is_holomorphic = False
        b_poly_vertices = np.nonzero(np.isin(self.boundary_vertices, poly_vertices))
        im_k = np.zeros_like(self.boundary_vertices, dtype=float)
        im_k[b_poly_vertices] = 2. * np.pi / len(poly_vertices)
        self._set_neumann(im_k)

    def setup_angles(self, boundary_angles: npt.ArrayLike) -> None:
        boundary_angles = self._pre_setup_angles(boundary_angles)
        im_k = np.roll(boundary_angles) - np.roll(boundary_angles, shift=-1)  # Clockwise boundary.
        self._set_neumann(im_k)

    def setup_lengths(self, boundary_lengths: npt.ArrayLike) -> None:
        elenu = boundary_lengths * np.log(boundary_lengths / self._boundary_edge_lengths)
        ub = (np.roll(elenu, shift=-1) + elenu) / (np.roll(boundary_lengths, shift=-1) + boundary_lengths)
        self._set_dirichlet(ub)

    def optimize(self) -> None:
        """Optimize conformal map.

        :return: None.
        """
        pass

    def _index_vertices(self, boundary: bool = False, cones: bool = False) -> IntArray:
        """Get vertices including/excluding boundary and cone singularities."""
        v_mask = np.ones(self.vertex_count, dtype=bool)
        v_mask[self.boundary_vertices] = boundary
        v_mask[self._cone_singularities] = cones
        return np.where(v_mask)[0]

    def _compute_scale_factors(self) -> FloatArray:
        """Compute scale factors assuming cone singularities have free angles."""
        ind_flat = self._index_vertices(boundary=False, cones=False)  # Vertices with zero target Gaussian curvature.
        u_flat = sparse_linalg.spsolve(self._A[ind_flat[:, np.newaxis], ind_flat], -self._angle_defects[ind_flat])
        u = np.zeros(self.vertex_count)
        u[ind_flat] = u_flat
        return u

    def _compute_cone_angles(self) -> FloatArray:
        """Compute cone angles."""
        u = self._compute_scale_factors()
        ind_non_cone = self._index_vertices(boundary=True, cones=False)
        Anc = self._A[np.array(self._cone_singularities)[:, np.newaxis], ind_non_cone]
        thetas = np.zeros(self.vertex_count)
        thetas[self._cone_singularities] = self._angle_defects[self._cone_singularities] + Anc @ u[ind_non_cone]
        for vid, angle in zip(self._cone_singularities, self._target_cone_angles):
            if angle is not None:
                thetas[vid] = angle
        return thetas

    def _add_singularities_by_id(self, vertex_ids: npt.ArrayLike, cone_angles: npt.ArrayLike,
                                 quantize: Optional[float]) -> None:
        vertex_ids, cone_angles = self._pre_add_singularities(vertex_ids, cone_angles)
        cone_count = len(vertex_ids)
        if quantize is not None:
            thetas = self._compute_cone_angles()
            self._target_cone_angles[-cone_count:] = quantize_value(thetas[self._cone_singularities[-cone_count:]],
                                                                    quantize)

    def _add_singularities_auto(self, count: int, cone_angles: npt.ArrayLike, quantize: Optional[float]) -> None:
        for i in range(count):
            picked = int(np.argmax(np.abs(self._compute_scale_factors())))
            self._add_singularities_by_id(picked, cone_angles[i], quantize)

    def layout(self, normalize: bool = True, start_vertices: List[int] = None,
               cut_tree: Dict[int, Set[int]] = None) -> TriangleMesh:
        if self._cone_singularities:
            if start_vertices is None and cut_tree is None:
                start_vertices, cut_tree = self.find_cuts(self._cone_singularities)
            cut_mesh = self.cut_faces(start_vertices, cut_tree)
            self._vmap_cut = np.zeros(cut_mesh.vertex_count, dtype=int)
            self._vmap_cut[cut_mesh.faces] = self.faces
            self._bff_cut = BFF(cut_mesh.vertices, cut_mesh.faces)  # Has unused solveA
            self._set_dirichlet(self._ub)
            self._bff_cut._is_holomorphic = False
            self._bff_cut.layout(normalize=normalize)
            self._image = self._bff_cut.image.copy(readonly=False)
        elif self.is_spherical:
            apex = self.vertex_count - 1
            vertices = self.vertices[:-1]  # apex == -1
            face_filter = np.ones(len(self.faces), dtype=bool)
            face_filter[self.get_vertex_faces(apex)] = False  # apex == -1
            faces = self.faces[face_filter]
            bff_cut = BFF(vertices, faces)
            image = bff_cut.layout(normalize=False)
            uv = mobius_normalize_layout(np.r_[image.vertices, np.full((1, 2), np.nan)])
            self._image = TriangleMesh.copy(self, copy_values=False, readonly=False)
            self._image.set_vertices(uv)
        else:
            uv = self._extend_curve(self._best_fit_curve())
            self._image = TriangleMesh.copy(self, copy_values=False, readonly=False)
            self._image.set_vertices(uv)

        if normalize and not self.is_spherical:
            self._image.normalize()
        self._image.set_readonly()
        return self._image

    def parameterize_uniform(self) -> TriangleMesh:
        self.setup_minimum_distortion()
        target_k = 2. * np.pi / len(self.boundary_vertices)

        uvb = None
        for _ in range(15):
            uvb = self._best_fit_curve()
            im_elen = np.linalg.norm(np.roll(uvb, shift=-1, axis=0) - uvb, axis=1)
            vmb = (im_elen + np.roll(im_elen, shift=1)) * 0.5
            im_k = 2. * np.pi * vmb / vmb.sum()
            im_k = (im_k + self._im_k) * 0.5  # Stabilize.
            # TODO: Implement a reasonable stopping condition.
            print(np.abs(self._im_k - target_k).max())
            self._set_neumann(im_k)

        self._is_holomorphic = False
        self._image = TriangleMesh(self._extend_curve(uvb), self.faces, readonly=True)
        return self._image

    def parameterize_curve(self, pcurve: ParametricCurve) -> TriangleMesh:
        self.setup_minimum_distortion()

        uvb = None
        for _ in range(15):
            uvb = self._best_fit_curve()
            im_elen = np.linalg.norm(np.roll(uvb, shift=-1, axis=0) - uvb, axis=1)
            s = np.cumsum(im_elen, axis=0)
            s /= s[-1]
            im_k = pcurve.compute_exterior_angles(s)
            # im_k = (im_k + self._im_k) * 0.5  # Stabilize.
            # TODO: Implement a reasonable stopping condition.
            self._set_neumann(im_k)

        self._is_holomorphic = False
        self._image = TriangleMesh(self._extend_curve(uvb), self.faces, readonly=True)
        return self._image

    def quasi_conformal_error(self) -> FloatArray:
        """Compute quasi-conformal error of the conformal map.

        :return: #F error-per-face.
        """
        return quasi_conformal_error(self.vertices[self.faces], self.image.vertices[self.image.faces])
