import pytest
import numpy as np
import jax.numpy as jnp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.collision import MRT
from xlb.grid import grid_factory
from xlb import DefaultConfig


def init_xlb_env(velocity_set):
    vel_set = velocity_set(precision_policy=xlb.PrecisionPolicy.FP32FP32, backend=ComputeBackend.JAX)
    xlb.init(
        default_precision_policy=xlb.PrecisionPolicy.FP32FP32,
        default_backend=ComputeBackend.JAX,
        velocity_set=vel_set,
    )

# MRT becomes BGK when M = I and S is a diagonal matrix with all values omega
@pytest.mark.parametrize(
    "dim,velocity_set,grid_shape,omega,M,S",
    [
        (2, xlb.velocity_set.D2Q9, (100, 100), 0.6, np.eye(9), 0.6*np.diag(np.ones(9,))),
        (2, xlb.velocity_set.D2Q9, (100, 100), 1.0, np.eye(9), 1.0*np.diag(np.ones(9,))),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 0.6, np.eye(19), 0.6*np.diag(np.ones(19,))),
        (3, xlb.velocity_set.D3Q19, (50, 50, 50), 1.0, np.eye(19), 1.0*np.diag(np.ones(19,))),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50), 0.6, np.eye(27), 0.6*np.diag(np.ones(27,))),
        (3, xlb.velocity_set.D3Q27, (50, 50, 50), 1.0, np.eye(27), 1.0*np.diag(np.ones(27,))),
    ],
)
def test_mrt_collision(dim, velocity_set, grid_shape, omega, M, S):
    init_xlb_env(velocity_set)
    my_grid = grid_factory(grid_shape)

    rho = my_grid.create_field(cardinality=1, fill_value=1.0)
    u = my_grid.create_field(cardinality=dim, fill_value=0.0)

    # Compute equilibrium
    compute_macro = QuadraticEquilibrium()
    f_eq = compute_macro(rho, u)

    # Compute collision

    compute_collision = MRT(M=M, S=S, velocity_set=DefaultConfig.velocity_set.q)

    f_orig = my_grid.create_field(cardinality=DefaultConfig.velocity_set.q)

    f_out = compute_collision(f_orig, f_eq, rho, u)

    Minv = np.linalg.inv(M)
    M = jnp.array(M)
    S = jnp.array(S)

    assert jnp.allclose(f_out, f_orig - jnp.tensordot(Minv, jnp.tensordot(S, jnp.tensordot(M, f_orig - f_eq, axes=(-1, 0)), axes=(-1, 0)), axes=(-1, 0)))
    assert jnp.allclose(f_out, f_orig - omega * (f_orig - f_eq))


if __name__ == "__main__":
    pytest.main()
