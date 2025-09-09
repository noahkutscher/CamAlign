from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad
from numpy_quaterinon import *

import cv2


def equisolid_linear(p_w: np.ndarray, t: np.ndarray, q: np.ndarray, alpha, delta: np.ndarray, resolution):
    z_comp = np.array([0, 0, 1])
    xy_comp = np.array([[1, 0 , 0], [0, 1, 0]])

    q = q / np.linalg.norm(q)

    q_inv = q * np.array([1, -1, -1, -1])
    
    p_q = point_to_true_qat(p_w)
    a = mul_quat(q, p_q)
    b = mul_quat(a, q_inv)
    p_rotated = true_quat_to_point(b)

    p_c = p_rotated + t
    p_c_xy = p_c @ xy_comp.T

    # the negative sign in this is needed as our points will always have negative z coordinates in camera space
    if(len(p_c.shape) > 1):
        d = np.sqrt((-np.sqrt((p_c @ z_comp)**2 / np.sum(p_c * p_c, axis=1)) + 1) / 2)[:, np.newaxis]
        # print(f"d:{p_rotated}")
        denom = alpha * np.sqrt(np.sum(p_c_xy * p_c_xy, axis=1))[:, np.newaxis]
    else:
        d = np.sqrt((-np.sqrt((p_c @ z_comp)**2 / np.dot(p_c, p_c)) + 1) / 2)
        # print(f"d:{p_rotated}")
        denom = alpha * np.sqrt(np.dot(p_c_xy, p_c_xy))

    nom = d * p_c_xy
    proj = nom / denom

    return proj * np.array([resolution[0], -resolution[0]]) + delta


def equisolid_linear_obj(params, p_w: np.ndarray, p_proj: np.ndarray, resolution, debug=False, DRAW_TARGET = None, DRAW_ESTIMATE = None, DEPTH_ESTIMATE = None):

    extract_t = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ])

    extract_q = np.array([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ])

    extract_a = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]])

    extract_d = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    t = np.dot(extract_t, params)
    q = np.dot(extract_q, params)
    alpha = np.dot(extract_a, params)
    d = np.dot(extract_d, params)

    est = equisolid_linear(p_w, t, q, alpha, d, resolution)
    if(debug):
        print(f"For Params: {params}")
        for i in range(len(p_w)):
            print(f"DEBUG: {p_w[i]} -> {est[i]}, gt: {p_proj[i]}")
            esttimat_depth = DEPTH_ESTIMATE[int(p_proj[i][1] + 540), int(p_proj[i][0] + 960)]    
            print(f"DEBUG: Depth Estimate: {esttimat_depth}, GT: {p_w[i]}")
            cv2.circle(DRAW_TARGET, (int(p_proj[i][0] + 960), int(p_proj[i][1] + 540)), 3, (0, 255, 0), -1)
            cv2.circle(DRAW_ESTIMATE, (int(est[i][0] + 960), int(est[i][1] + 540)), 3, (255, 0, 0), -1)


        cv2.imshow("Target", DRAW_TARGET)
        cv2.imshow("Estimate", DRAW_ESTIMATE)
        cv2.waitKey(0)
        cv2.destroyWindow("Target")
        cv2.destroyWindow("Estimate")
    
    dist = p_proj - est
    if(len(dist.shape) > 1):
        return np.mean(np.sqrt(np.sum(dist * dist, axis=1)))
    else:
        return np.mean(np.sqrt(np.sum(dist * dist)))


def RunCameraOptimization(initial_guess, optimizer_func="L-BFGS-B", use_constraint=False, camera_resolution = (1920, 1080), point_database = {}):
    coord_world = []
    coord_projected = []
    center = np.array([camera_resolution[0] / 2, camera_resolution[1] / 2])
    offset = np.array([0.5, 0.5])
    for k in point_database:
        point = point_database[k]
        coord_projected.append(point[0] - center + offset)
        coord_world.append(point[1][:3])

    # batch the inputs together for easier compute
    coord_world = np.array(coord_world)
    coord_projected = np.array(coord_projected)

    errors = []

    print(f"DEBUG COORDS: {coord_world}")
    print(f"DEBUG COORDS Proj: {coord_projected}")

    if not optimizer_func:
        bfgs_params = initial_guess
        error = equisolid_linear_obj(bfgs_params, coord_world, coord_projected, camera_resolution)
        errors.append(error)


    else:
        def callback(res):
            error = equisolid_linear_obj(res, coord_world, coord_projected, camera_resolution)
            errors.append(error)

        # print("Running Optimization")
        # the minimize function could also be used with the functions mentioned in https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab to compare as 1st degree functions
        jac = grad(equisolid_linear_obj)
        hes = grad(jac)
        if use_constraint:
            def quaternion_norm_contraint(x):
                return np.linalg.norm(x[3:7]) - 1
            constraint = {'type':'eq', 'fun': quaternion_norm_contraint}

            optim_result = minimize(equisolid_linear_obj, initial_guess, args=(coord_world, coord_projected, camera_resolution), method=optimizer_func, jac=jac, hess=hes, callback=callback, constraints=[constraint])
        else:
            optim_result = minimize(equisolid_linear_obj, initial_guess, args=(coord_world, coord_projected, camera_resolution), method=optimizer_func, jac=jac, hess=hes, callback=callback)

        bfgs_params = optim_result.x
        error = equisolid_linear_obj(bfgs_params, coord_world, coord_projected, camera_resolution)


    return bfgs_params, error, errors