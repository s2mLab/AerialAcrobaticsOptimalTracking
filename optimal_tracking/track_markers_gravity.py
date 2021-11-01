import biorbd_casadi as biorbd
import numpy as np
import ezc3d
import time
from casadi import MX, Function
import pickle
import os
from load_data_filename import load_data_filename
from x_bounds import x_bounds
from adjust_number_shooting_points import adjust_number_shooting_points
from reorder_markers import reorder_markers
from adjust_Kalman import shift_by_2pi

from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    Bounds,
    InitialGuessList,
    InitialGuess,
    InterpolationType,
    Solver,
    OdeSolver,
    Node,
    ParameterList,
)


def rotating_gravity(biorbd_model, value):
    # The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    # modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it
    # `biorbd_model` and `value` are mandatory. The former is the actual model to modify, the latter is the casadi.MX
    # used to modify it,  the size of which decribed by the value `size` in the parameter definition.
    # The rest of the parameter are defined by the user in the parameter
    gravity = biorbd_model.getGravity()
    gravity.applyRT(
        biorbd.RotoTrans.combineRotAndTrans(biorbd.Rotation.fromEulerAngles(value, 'zx'), biorbd.Vector3d()))
    biorbd_model.setGravity(gravity)


def inverse_dynamics(biorbd_model, q_ref, qd_ref, qdd_ref):
    q = MX.sym("Q", biorbd_model.nbQ(), 1)
    qdot = MX.sym("Qdot", biorbd_model.nbQdot(), 1)
    qddot = MX.sym("Qddot", biorbd_model.nbQddot(), 1)
    id = biorbd.to_casadi_func("id", biorbd_model.InverseDynamics, q, qdot, qddot)

    return id(q_ref, qd_ref, qdd_ref)[:, :-1]



def prepare_ocp(biorbd_model, final_time, number_shooting_points, markers_ref, q_init, qdot_init, tau_init, xmin, xmax, min_g, max_g, min_torque_diff=False):
    torque_min, torque_max = -300, 300
    n_q = biorbd_model.nbQ()
    n_tau = biorbd_model.nbGeneralizedTorque()

    # Add objective functions
    state_ref = np.concatenate((q_init, qdot_init))
    objective_functions = ObjectiveList()

    # Tracking term
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_MARKERS, node=Node.ALL, weight=1, target=markers_ref)

    # Regularization terms
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1e-7)
    if min_torque_diff:
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1e-5)

    # Extra regularization terms
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, node=Node.ALL, weight=1e-5, target=state_ref)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL, weight=1e-5, index=range(6, n_q))
    # objective_functions.add(ObjectiveFcn.Lagrange.TRACK_CONTROL, key="tau", weight=1e-7, target=tau_init)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    X_bounds = BoundsList()
    X_bounds.add(min_bound=xmin, max_bound=xmax)

    # Initial guess
    # Option to set initial guess to zero if it is abnormal
    # q_init = np.zeros(q_init.shape)
    # qdot_init = np.zeros(qdot_init.shape)
    X_init = InitialGuessList()
    X_init.add(np.concatenate([q_init, qdot_init]), interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    U_bounds = BoundsList()
    U_bounds.add(min_bound=[torque_min] * n_tau, max_bound=[torque_max] * n_tau)
    # The root is in free-fall, so it has no control
    U_bounds[0].min[:6, :] = 0
    U_bounds[0].max[:6, :] = 0

    U_init = InitialGuessList()
    # Option to set initial guess to zero if it is abnormal
    # tau_init = np.zeros(tau_init.shape)
    U_init.add(tau_init, interpolation=InterpolationType.EACH_FRAME)

    # Define the parameter to optimize
    parameters = ParameterList()
    bound_gravity = Bounds(min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
    initial_gravity_orientation = InitialGuess([0, 0])
    parameters.add(
        parameter_name="gravity_angle",  # The name of the parameter
        function=rotating_gravity,  # The function that modifies the biorbd model
        bounds=bound_gravity,  # The bounds
        initial_guess=initial_gravity_orientation,  # The initial guess
        size=2,  # The number of elements this particular parameter vector has
    )

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        ode_solver=OdeSolver.RK4(n_integration_steps=4),
        parameters=parameters,
        n_threads=4,
    )


if __name__ == "__main__":
    start = time.time()
    subject = 'DoCi'
    number_shooting_points = 100
    trial = '44_3'
    print('Subject: ', subject, ', Trial: ', trial)

    # Choose between track_q or EKF for an initial guess
    initial_guess_track_Q = False

    # Add trial to this list if adding a regularization term on the control derivative is judged necessary
    trial_needing_min_torque_diff = {
                                     # 'DoCi': ['822'],
                                     }

    min_torque_diff = False
    if subject in trial_needing_min_torque_diff.keys():
        if trial in trial_needing_min_torque_diff[subject]:
            min_torque_diff = True

    data_path = 'data/' + subject + '/'

    data_filename = load_data_filename(subject, trial)
    model_name = data_filename['model']
    c3d_name = data_filename['c3d']
    frames = data_filename['frames']

    biorbd_model = biorbd.Model(data_path + model_name)
    c3d = ezc3d.c3d(data_path + c3d_name)

    initial_gravity = biorbd.Vector3d(0, 0, -9.80639)
    biorbd_model.setGravity(initial_gravity)

    # --- Adjust number of shooting points --- #
    adjusted_number_shooting_points, step_size = adjust_number_shooting_points(number_shooting_points, frames)
    print('Adjusted number of shooting points: ', adjusted_number_shooting_points)
    print('Node step size: ', step_size)

    frequency = c3d['header']['points']['frame_rate']
    duration = len(frames) / frequency

    # --- Load initial guess --- #
    if initial_guess_track_Q:
        track_q_filename = 'solutions/track_q/' + os.path.splitext(c3d_name)[0] + '_N' + str(adjusted_number_shooting_points) + ".bo"
        ocp_track_q, sol_track_q = OptimalControlProgram.load(track_q_filename)
        q_track_q = sol_track_q.states['q']
        qdot_track_q = sol_track_q.states['qdot']
        tau_track_q = sol_track_q.controls['tau']

        q_ref = q_track_q
        qdot_ref = qdot_track_q
        tau_ref = tau_track_q[:, :-1]
    else:
        load_path = 'solutions/EKF/'
        load_name = load_path + os.path.splitext(c3d_name)[0] + ".pkl"
        with open(load_name, 'rb') as handle:
            EKF = pickle.load(handle)
        q_EKF = shift_by_2pi(biorbd_model, EKF['q'][:, ::step_size])
        qdot_EKF = EKF['qd'][:, ::step_size]
        qddot_EKF = EKF['qdd'][:, ::step_size]

        tau_EKF = inverse_dynamics(biorbd_model, q_EKF, qdot_EKF, qddot_EKF)

        q_ref = q_EKF
        qdot_ref = qdot_EKF
        tau_ref = tau_EKF

    xmin, xmax = x_bounds(biorbd_model)

    # Organize the markers in the same order as in the model
    markers_reordered, _ = reorder_markers(biorbd_model, c3d, frames, step_size)

    ocp = prepare_ocp(
        biorbd_model=biorbd_model, final_time=duration, number_shooting_points=adjusted_number_shooting_points,
        markers_ref=markers_reordered, q_init=q_ref, qdot_init=qdot_ref, tau_init=tau_ref,
        xmin=xmin, xmax=xmax, min_torque_diff=min_torque_diff,
        min_g=[0, -np.pi/16], max_g=[2*np.pi, np.pi/16],
    )

    # --- Solve the program --- #
    options = {"max_iter": 3000, "tol": 1e-4, "constr_viol_tol": 1e-2, "linear_solver": "ma57"}
    sol = ocp.solve(solver=Solver.IPOPT, solver_options=options, show_online_optim=False)

    # --- Get the results --- #
    q_sol = sol.states['q']
    qdot_sol = sol.states['qdot']
    tau_sol = sol.controls['tau']
    params = sol.parameters

    # --- Save --- #
    save_path = 'solutions/track_markers_gravity/'
    save_name = save_path + os.path.splitext(c3d_name)[0] + "_N" + str(adjusted_number_shooting_points)
    ocp.save(sol, save_name + ".bo")

    biorbd_model = biorbd.Model(data_path + model_name)
    biorbd_model.setGravity(initial_gravity)
    angle = params["gravity_angle"].squeeze()
    rotating_gravity(biorbd_model, angle)
    get_gravity = Function('get_gravity', [], [biorbd_model.getGravity().to_mx()], [], ['gravity'])
    gravity = get_gravity()['gravity'].full().squeeze()

    save_variables_name = save_name + ".pkl"
    with open(save_variables_name, 'wb') as handle:
        pickle.dump({'mocap': markers_reordered, 'duration': duration, 'frames': frames, 'step_size': step_size,
                     'q': q_sol, 'qdot': qdot_sol, 'tau': tau_sol, 'gravity': gravity, 'gravity_angle': angle},
                    handle, protocol=3)

    print('Number of shooting points: ', adjusted_number_shooting_points)
    print('Gravity rotation: ', angle/np.pi*180)
    print('Gravity: ', gravity)

    stop = time.time()
    print('Runtime: ', stop - start)
