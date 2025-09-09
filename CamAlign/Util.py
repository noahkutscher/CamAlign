from pathlib import Path
import math
import json
from scipy.spatial import transform
import autograd.numpy as np
from numpy_quaterinon import *


class CameraState:
    loc = np.array([0, 0, 0])
    loc_inv = np.array([0, 0, 0])
    rot_euler = np.array([0, 0, 0])
    rot_quat = np.array([1, 0, 0, 0])
    alpha = 0.9

    res_width = 1920
    res_height = 1080
    point_database = {}

    # DONT ACCESS
    s = 0
    f = 0

def LoadSample(save_folder):
    if not (save_folder / "points.json").exists():
        print("Error: No PPosition Buffer Found")
        return

    with open(save_folder / "points.json", "r") as f:
        load_dict = json.load(f)

    state = CameraState()

    if save_folder.name == "000005":
        state.res_width = 2688
        state.res_height = 1520

    state.point_database.clear()
    running_idx = 0
    for k in load_dict["points"]:
        num = int(k.split("_")[1])
        running_idx = max(running_idx, num+1)
        state.point_database[k] = [
            np.array(load_dict["points"][k][0]),
            np.array(load_dict["points"][k][1])
        ]


    state.loc = np.array(load_dict["info"]["loc"])
    state.rot_euler = np.array(load_dict["info"]["rot"]) * 180 / math.pi
    state.rot_quat = transform.Rotation.from_euler('xyz', state.rot_euler, degrees=True).as_quat(scalar_first=True)
    
    state.loc_inv = InverseLocation(state.loc, state.rot_euler)


    foc = load_dict["info"]["f"]
    s = load_dict["info"]["s"]
    state.alpha = s / (2 * foc)

    # use this for debugging
    state.s = s
    state.f = foc
    ########

    return state

def InverseLocation(location, rotation):
    rot_mat = transform.Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
    c2w = np.identity(4)
    c2w[:3, :3] = rot_mat
    c2w[:3, 3] = location
    w2c = np.linalg.inv(c2w)
    return w2c[:3, 3]



def LoadSampleFromDict(load_dict):
    state = CameraState()

    state.point_database.clear()
    running_idx = 0

    if "points" in load_dict:
        for k in load_dict["points"]:
            num = int(k.split("_")[1])
            running_idx = max(running_idx, num+1)
            state.point_database[k] = [
                np.array(load_dict["points"][k][0]),
                np.array(load_dict["points"][k][1])
            ]


    state.loc = np.array(load_dict["info"]["loc"])
    state.rot_euler = np.array(load_dict["info"]["rot"]) * 180 / math.pi
    state.rot_quat = transform.Rotation.from_euler('xyz', state.rot_euler, degrees=True).as_quat(scalar_first=True)

    rot_mat = transform.Rotation.from_euler('xyz', state.rot_euler, degrees=True).as_matrix()
    c2w = np.identity(4)
    c2w[:3, :3] = rot_mat
    c2w[:3, 3] = state.loc

    w2c = np.linalg.inv(c2w)
    state.loc_inv = w2c[:3, 3]
    foc = load_dict["info"]["f"]
    s = load_dict["info"]["s"]
    state.alpha = s / (2 * foc)

    # use this for debugging
    state.s = s
    state.f = foc
    ########

    return state




def SaveSample(save_folder: Path, cinf: CameraState):
    save_folder.mkdir(exist_ok=True, parents=True)

    save_dict = {}

    save_dict["info"] = {
        "loc": list(cinf.loc),
        "rot": list((cinf.rot_euler * math.pi) / 180),
        "rot_info": "XYZ",
        "f": cinf.f,
        "s": cinf.s,
    }
    
    save_dict["points"] = {

    }
    
    for k in cinf.point_database:
        save_dict["points"][k] = [
            cinf.point_database[k][0].tolist(),
            cinf.point_database[k][1].tolist()
        ]

    with open(save_folder / "points.json", "w") as f:
        json.dump(save_dict, f)