import autograd.numpy as np

#region quaternion logic 
q_x = np.array([0, 1, 0, 0])
q_y = np.array([0, 0, 1, 0])
q_z = np.array([0, 0, 0, 1])
q_w = np.array([1, 0, 0, 0])
qm_x = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, -1],
    [0, 0, 1, 0]
])

qm_y = np.array([
    [0, 0, -1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, -1, 0, 0]
])

qm_z = np.array([
    [0, 0, 0, -1],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0]
])

qm_w = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


def point_to_true_qat(p: np.ndarray):
    a = np.array([
        [0, 0, 0],
        [1, 0, 0], 
        [0, 1, 0],
        [0, 0, 1]
    ])
    # use matmul instead of dot product so it works with batched values for autograd
    return p @ a.T

def true_quat_to_point(p: np.ndarray):
    a = np.array([
        [0, 1, 0, 0], 
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # use matmul instead of dot product so it works with batched values for autograd
    return p @ a.T

def quat_get_dim_mat(q, extr, mat):
    q_ex = q @ extr
    if(len(q_ex.shape) > 0):
        q_ex = q_ex[:, np.newaxis, np.newaxis]
        return np.multiply(q_ex, mat).transpose([0, 2, 1])

    return np.multiply(q_ex, mat).T

def quat_to_mat(q: np.ndarray):
    qm_ex = quat_get_dim_mat(q, q_x, qm_x)
    qm_ey = quat_get_dim_mat(q, q_y, qm_y)
    qm_ez = quat_get_dim_mat(q, q_z, qm_z)
    qm_ew = quat_get_dim_mat(q, q_w, qm_w)

    ret = qm_ex + qm_ey + qm_ez + qm_ew
    return ret


def mul_quat(p, q):
    return q @ quat_to_mat(p)

def quaternion_conjugate(q):
    return q * np.array([1, -1, -1, -1])

def quaternion_from_axis_angle(axis, angle):
    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    return np.array([w, *xyz])

# this might be non differentiable
def quaternion_angle(p, q):
    p = p / np.linalg.norm(p)
    q = q / np.linalg.norm(q)

    # Relative rotation
    q1_inv = quaternion_conjugate(p)  # Since quaternions are unit, inverse == conjugate
    q_relative = mul_quat(q1_inv, q)

    angle = 2 * np.arccos(np.clip(q_relative[0], -1.0, 1.0))  # q_relative[0] is w
    return angle


def quaternion_to_euler(q):
    """
    Convert a quaternion [w, x, y, z] to Euler angles [x, y, z] in radians.
    Rotation order is XYZ.
    """
    w, x, y, z = q

    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    # Pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # Clamp to avoid invalid values
    pitch_y = np.arcsin(t2)

    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.degrees([roll_x, pitch_y, yaw_z])


# Example usage:
# q1 = np.array([0.7071, 0.0, 0.7071, 0.0])  # 90 deg around Y
# q2 = np.array([1.0, 0.0, 0.0, 0.0])        # Identity rotation

# angle_rad = quaternion_angle(q1, q2)
# angle_deg = np.degrees(angle_rad)

# print(f"Angle (radians): {angle_rad}")
# print(f"Angle (degrees): {angle_deg}")