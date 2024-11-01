import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq, check_bc_overlaps
from xlb.operator.boundary_masker import IndicesBoundaryMasker
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import HalfwayBounceBackBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.utils import save_fields_vtk, save_image
import xlb.velocity_set
import warp as wp
import jax.numpy as jnp
import numpy as np


class LidDrivenCavity2D:
    def __init__(self, prescribed_vel, grid_shape, velocity_set, backend, precision_policy):
        # initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        self.grid_shape = grid_shape
        self.velocity_set = velocity_set
        self.backend = backend
        self.precision_policy = precision_policy
        self.grid, self.f_0, self.f_1, self.missing_mask, self.bc_mask = create_nse_fields(grid_shape)
        self.stepper = None
        self.boundary_conditions = []
        self.prescribed_vel = prescribed_vel

        # Setup the simulation BC, its initial conditions, and the stepper
        self._setup()

    def _setup(self):
        self.setup_boundary_conditions()
        self.setup_boundary_masker()
        self.initialize_fields()
        self.setup_stepper()

    def define_boundary_indices(self):
        box = self.grid.bounding_box_indices()
        box_no_edge = self.grid.bounding_box_indices(remove_edges=True)
        lid = box_no_edge["top"]
        walls = [box["bottom"][i] + box["left"][i] + box["right"][i] for i in range(self.velocity_set.d)]
        walls = np.unique(np.array(walls), axis=-1).tolist()
        return lid, walls

    def setup_boundary_conditions(self):
        lid, walls = self.define_boundary_indices()
        bc_top = EquilibriumBC(rho=1.0, u=(self.prescribed_vel, 0.0), indices=lid)
        bc_walls = HalfwayBounceBackBC(indices=walls)
        self.boundary_conditions = [bc_walls, bc_top]

    def setup_boundary_masker(self):
        # check boundary condition list for duplicate indices before creating bc mask
        check_bc_overlaps(self.boundary_conditions, self.velocity_set.d, self.backend)
        indices_boundary_masker = IndicesBoundaryMasker(
            velocity_set=self.velocity_set,
            precision_policy=self.precision_policy,
            compute_backend=self.backend,
        )
        self.bc_mask, self.missing_mask = indices_boundary_masker(self.boundary_conditions, self.bc_mask, self.missing_mask)

    def initialize_fields(self):
        self.f_0 = initialize_eq(self.f_0, self.grid, self.velocity_set, self.precision_policy, self.backend)

    def setup_stepper(self):
        self.stepper = IncompressibleNavierStokesStepper(omega=0.0, M=M, S=S, boundary_conditions=self.boundary_conditions, collision_type="MRT")

    def run(self, num_steps, post_process_interval=100):
        for i in range(num_steps):
            self.f_0, self.f_1 = self.stepper(self.f_0, self.f_1, self.bc_mask, self.missing_mask, i)
            self.f_0, self.f_1 = self.f_1, self.f_0

            if i % post_process_interval == 0 or i == num_steps - 1:
                self.post_process(i)

    def post_process(self, i):
        # Write the results. We'll use JAX backend for the post-processing
        if not isinstance(self.f_0, jnp.ndarray):
            # If the backend is warp, we need to drop the last dimension added by warp for 2D simulations
            f_0 = wp.to_jax(self.f_0)[..., 0]
        else:
            f_0 = self.f_0

        macro = Macroscopic(
            compute_backend=ComputeBackend.JAX,
            precision_policy=self.precision_policy,
            velocity_set=xlb.velocity_set.D2Q9(precision_policy=self.precision_policy, backend=ComputeBackend.JAX),
        )

        rho, u = macro(f_0)

        # remove boundary cells
        rho = rho[:, 1:-1, 1:-1]
        u = u[:, 1:-1, 1:-1]
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5

        fields = {"rho": rho[0], "u_x": u[0], "u_y": u[1], "u_magnitude": u_magnitude}

        save_fields_vtk(fields, timestep=i, prefix="lid_driven_cavity")
        save_image(fields["u_magnitude"], timestep=i, prefix="lid_driven_cavity")


if __name__ == "__main__":
    # Running the simulation
    grid_size = 500
    grid_shape = (grid_size, grid_size)
    backend = ComputeBackend.JAX
    precision_policy = PrecisionPolicy.FP32FP32

    velocity_set = xlb.velocity_set.D2Q9(precision_policy=precision_policy, backend=backend)

    # Setting fluid viscosity and relaxation parameter.
    Re = 200.0
    prescribed_vel = 0.05
    clength = grid_shape[0] - 1
    visc = prescribed_vel * clength / Re

    # Mohammed Jami, Fayçal Moufekkir, Ahmed Mezrhab, Jean Pierre Fontaine, M'hamed Bouzidi
    # "New thermal MRT lattice Boltzmann method for simulations of convective flows"
    # https://www.sciencedirect.com/science/article/pii/S1290072915002720?via%3Dihub

    # The lattice velocities are defined in this order:
    cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    cy = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    c_ = np.array(tuple(zip(cx, cy))).reshape((-1, 2))
    c = np.zeros((9, 2))
    c[:, 0] = velocity_set.c[0]
    c[:, 1] = velocity_set.c[1]
    idx = []

    # Comparing order between c and c_
    for i in range(velocity_set.q):
        for j in range(velocity_set.q):
            if np.allclose(c_[i, :], c[j, :]):
                idx.append(j)

    M_ = np.zeros((velocity_set.q, velocity_set.q))
    M_[0, :] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    M_[1, :] = [-4, -1, -1, -1, -1, 2, 2, 2, 2]
    M_[2, :] = [4, -2, -2, -2, -2, 1, 1, 1, 1]
    M_[3, :] = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    M_[4, :] = [0, -2, 0, 2, 0, 1, -1, -1, 1]
    M_[5, :] = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    M_[6, :] = [0, 0, -2, 0, 2, 1, 1, -1, -1]
    M_[7, :] = [0, 1, -1, 1, -1, 0, 0, 0, 0]
    M_[8, :] = [0, 0, 0, 0, 0, 1, -1, 1, -1]

    # M defined in the order of velocity_set
    M = np.zeros_like(M_)
    for i in range(len(idx)):
        M[i, :] = M_[idx[i], :]

    s_0 = 0.0
    s_1 = 1.4
    s_2 = s_1
    s_3 = 0.0
    s_4 = 1.2
    s_5 = 0.0
    s_6 = s_4
    s_7 = 1 / (3.0 * visc + 0.5)
    s_8 = s_7

    d = [s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8]
    S = np.zeros((9, 9))

    # S defined in the order of velocity_set
    for i in range(len(idx)):
        S[i, i] = d[idx[i]]

    simulation = LidDrivenCavity2D(prescribed_vel, grid_shape, velocity_set, backend, precision_policy)
    simulation.run(num_steps=50000, post_process_interval=1000)
