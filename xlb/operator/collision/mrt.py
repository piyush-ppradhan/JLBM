import numpy as np
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from functools import partial


class MRT(Operator):
    """
    MRT collision operator for LBM.

    Note:
    The sequence of velocity directions in `velocity_set` defines the order of rows in the transformation matrix `M` and the diagonal elements in
    collision matrix `S`. Since different sources may specify `M` and `S` based on their chosen discrete velocity order, you’ll need to reorder the
    rows of `M` and the diagonal elements of `S` to match the sequence set in `velocity_set`.
    """

    def __init__(self, M: np.ndarray, S: np.ndarray, velocity_set, precision_policy=None, compute_backend=None):
        super().__init__(velocity_set, precision_policy, compute_backend)
        self.M = M
        self.Minv = np.linalg.inv(M)
        self.S = S

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(self, f: jnp.ndarray, feq: jnp.ndarray, rho, u):
        self.M = jnp.array(self.M, self.compute_dtype)
        self.Minv = jnp.array(self.Minv, self.compute_dtype)
        self.S = jnp.array(self.S, self.compute_dtype)
        fneq = f - feq
        fout = f - jnp.tensordot(self.Minv, jnp.tensordot(self.S, jnp.tensordot(self.M, fneq, axes=(-1, 0)), axes=(-1, 0)), axes=(-1, 0))
        return fout

    def _construct_warp(self):
        # TODO
        # BGK code as placeholder
        _w = self.velocity_set.w
        _omega = wp.constant(self.compute_dtype(0.0))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(f: Any, feq: Any, rho: Any, u: Any):
            fneq = f - feq
            fout = f - _omega * fneq
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, rho, u)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = self.store_dtype(_fout[l])

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, fout, rho, u):
        # TODO
        # BGK code as placeholder
        raise NotImplementedError("Warp backend for MRT collision has not been implemented yet.")