import numpy as np
import math

def start_to_current(T_w_start: np.ndarray, T_w_cur: np.ndarray, points_start: np.ndarray) -> np.ndarray:
    """
    points_start: (N,2) in START frame
    returns:      (N,2) in CURRENT frame
    p^c = (T^w_c)^-1 * T^w_s * p^s
    """
    pts = np.asarray(points_start, dtype=np.float64)
    N = pts.shape[0]
    pts_h = np.ones((3, N), dtype=np.float64)
    pts_h[0, :] = pts[:, 0]
    pts_h[1, :] = pts[:, 1]

    T_c_s = np.linalg.inv(T_w_cur) @ T_w_start  # 3x3
    pts_c = (T_c_s @ pts_h)[:2, :].T  # (N,2)

    return pts_c


def odom_to_robot(config, x_odom, y_odom):
    # print(x_odom.shape[0])
    x_rob_odom_list = np.asarray([config.x for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([config.y for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list) * math.cos(config.th) + (y_odom - y_rob_odom_list) * math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list) * math.sin(config.th) + (y_odom - y_rob_odom_list) * math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob