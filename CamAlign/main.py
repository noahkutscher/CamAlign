import sys
print(sys.executable)

import json

import bpy
import bpy_extras
import numpy as np
import cv2
import threading
from threading import Lock
from pathlib import Path
import traceback
import math
import queue

from Optimizer import RunCameraOptimization, equisolid_linear_obj
from Util import *
from numpy_quaterinon import *
from FeatureMatcher import RunLightGlueMatcher

from scipy.optimize import minimize

from ZoomableWindow.ZoomableWindow import ZoomableWindow

class EditCommand:
    command = ""
    object_name = ""
    location = np.array([0, 0, 0])

db_lock = Lock()

point_database = {

}

COM_EDIT2BLENDER = queue.Queue(200)
COM_BLENDER2EDIT = queue.Queue(200)

EDITOR_OPEN = False
DATABASE_DIRTY = False

##############################################
############ DEBUG ################
##############################################



def calculate_equisolid_with_position(p: np.ndarray, px, py, pz, rx, ry, rz, foc: float, s_w):
    # project point into camera space
    c2w = np.identity(4)
    rot = transform.Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
    c2w[:3, :3] = rot
    c2w[:3, 3] = [px, py, pz]

    w2c = np.linalg.inv(c2w)

    p = w2c @ p

    # get the pixel size in mm so we can convert the sensor point back to pixels
    pixel_size = s_w / 1920

    l = (p[0]**2 + p[1]**2)**(1/2)
    # attention... atan assumes a positive axis direction, but we look along -z so we need to calculate accordingly
    theta = math.atan2(l, -p[2])

    # Equisolid projection
    ip_dir = np.array([p[0], p[1]])
    ip_dir = ip_dir / np.linalg.norm(ip_dir)
    
    r = 2.0 * foc * math.sin(theta / 2)

    u = r * (ip_dir[0])
    v = r * (ip_dir[1])

    x = u / pixel_size
    y = -v / pixel_size

    return np.array([x, y])

def calculate_equisolid_pose_obj(params, p_in, p_true, debug = False):
    foc, px, py, pz, rx, ry, rz, s_w = params
    sum_err = 0
    for i in range(len(p_in)):
        p_est = calculate_equisolid_with_position(p_in[i], px, py, pz, rx, ry, rz, foc, s_w)
        err = np.linalg.norm(p_est - p_true[i])
        sum_err += err

        if debug:
             print(f"DEBUG: {p_in[i]} -> {p_est}, gt: {p_true[i]}")

    return sum_err / len(p_in)

# as initial guess use the default information
def optimize_camera_parameters(initial_guess = [10.5, 0, 0, 0, 0, 0, 0, 36]):
    global point_database
    coord_world = []
    coord_projected = []
    center = np.array([1920 / 2, 1080 / 2])
    offset = np.array([0.5, 0.5])

    with db_lock:
        for k in point_database:
            point = point_database[k]
            coord_projected.append(point[0] - center + offset)
            coord_world.append(point[1])

    # the minimize function could also be used with the functions mentioned in https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab to compare as 1st degree functions
    optim_result = minimize(calculate_equisolid_pose_obj, initial_guess, args=(coord_world, coord_projected), method="L-BFGS-B")
    bfgs_params = optim_result.x
    print(f"Params: {bfgs_params}")
    error = calculate_equisolid_pose_obj(bfgs_params, coord_world, coord_projected, True)
    print(f"Error: {error}")

    print("--------------------------")
    print(f"Testing BFGS guess")
    print("--------------------------")

    guess = calculate_equisolid_with_position(coord_world[0],
        bfgs_params[1],
        bfgs_params[2],
        bfgs_params[3],
        bfgs_params[4],
        bfgs_params[5],
        bfgs_params[6],
        bfgs_params[0],
        bfgs_params[7],
    )
    err = np.linalg.norm(guess - coord_projected[0])
    print(f"guessed {guess}")
    print(f"from {coord_projected[0][0]}, {coord_projected[0][1]}")

    print(f"Error: {err}")

    return bfgs_params

##############################################
############ Feature Matching ################
##############################################
def populatePointDatabase():
    global point_database, COM_EDIT2BLENDER, DATABASE_DIRTY

    db = RunLightGlueMatcher(TARGET_FRAME, CURRENT_RENDER, CURRENT_POSITION_BUFFER)
    with db_lock:
        point_database = db
        DATABASE_DIRTY = True

    for name in point_database:
        com = EditCommand()
        com.command = "ADD"
        com.location = point_database[name][1][:3]
        com.object_name = name
        COM_EDIT2BLENDER.put(com)

##############################################
############### Render UI ###################
##############################################

RENDER_DONE = False
RENDER_PROG = 0.0

CURRENT_RENDER = None
CURRENT_POSITION_BUFFER = None

TARGET_FRAME = None
TARGET_EDITABLE = None
TARGET_POSITIONS_DEBUG = None

def constructRenderGraph():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    rl = tree.nodes.new('CompositorNodeRLayers')      
    rl.location = 185,285

    viewerNode = tree.nodes.new('CompositorNodeViewer')  
    viewerNode.label = "OutputNode" 
    viewerNode.location = 750,300
    viewerNode.use_alpha = False

    links.new(rl.outputs["Image"], viewerNode.inputs["Image"])  # link Image output to Viewer input

    return rl, viewerNode

def pass_to_viewernode(pass_name: str, renderLayers: bpy.types.Node, viewerNode: bpy.types.Node):
    tree = bpy.context.scene.node_tree
    tree.links.new(renderLayers.outputs[pass_name], viewerNode.inputs["Image"])
    
def get_pass_img(passname: str, n_channels: int, dtype: str, renderLayers, viewerNode):
    print(f"Retrieving for Pass: {passname}")
    pass_to_viewernode(passname, renderLayers, viewerNode)

    print("Starting Render...")
    bpy.ops.render.render(write_still=True)
    print("Finished Render")

    render_result = bpy.data.images["Viewer Node"]

    width = render_result.size[0]
    height = render_result.size[1] 
    raw_pixels = np.array(render_result.pixels)

    img = raw_pixels.reshape((height, width, 4))
    img = img[:, :, :3]
    img = cv2.flip(img, 0)

    return img

def setup_renderer(filename, camera):
    output_path = Path(__file__).parent / filename

    bpy.context.view_layer.use_pass_position = True

    scene = bpy.context.scene
    scene.render.filepath = str(output_path)
    scene.render.engine = 'CYCLES'  
    scene.camera = camera
    scene.frame_set(1)

    return output_path

def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):

    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

def run_render():
    global RENDER_DONE, RENDER_PROG, CURRENT_RENDER, CURRENT_POSITION_BUFFER, TARGET_EDITABLE, TARGET_POSITIONS_DEBUG, TARGET_FRAME
    RENDER_DONE = False

    # render the prior camera
    if 'Camera_Prior' in bpy.data.objects:
        render_path = setup_renderer("color.png", bpy.data.objects['Camera_Prior'])
        renderLayers, viewerNode = constructRenderGraph()
        
        CURRENT_POSITION_BUFFER = get_pass_img("Position", 3, np.float32, renderLayers, viewerNode)
        RENDER_PROG = 0.5
        CURRENT_RENDER = cv2.imread(str(render_path))

        TARGET_EDITABLE = CURRENT_RENDER.copy()

        # render the target
        if 'Camera' in bpy.data.objects:
            render_path = setup_renderer("color_target.png", bpy.data.objects['Camera'])
            renderLayers, viewerNode = constructRenderGraph()
            
            TARGET_POSITIONS_DEBUG = get_pass_img("Position", 3, np.float32, renderLayers, viewerNode)
            TARGET_FRAME = cv2.imread(str(render_path))
            TARGET_EDITABLE = TARGET_FRAME.copy()

    else:
        ShowMessageBox("You need to create a \'Camera_Prior\' object to define the initial guess.", "No Initial Camera Found!", 'ERROR')
        
    RENDER_PROG = 1.0
    RENDER_DONE = True



##############################################
############### CV2 UI ###################
##############################################

WINDOW_TITLE = "Editor"

running_idx = 0
selected_point = None

EDITOR_WINDOW: ZoomableWindow = None

def redraw():
    global point_database, EDITOR_WINDOW
    CURRENT_EDITABLE = TARGET_FRAME.copy()

    if EDITOR_WINDOW is None:
        EDITOR_WINDOW = ZoomableWindow(WINDOW_TITLE, CURRENT_EDITABLE)


    with db_lock:
        for k in point_database:
            point = point_database[k][0]
            if k == selected_point:
                cv2.circle(CURRENT_EDITABLE, (point[0], point[1]), 3, (255, 0, 0))
            else:
                cv2.circle(CURRENT_EDITABLE, (point[0], point[1]), 3, (0, 0, 255))

        EDITOR_WINDOW.image = CURRENT_EDITABLE
        EDITOR_WINDOW.show()

def spawn_click(x, y, flags, param):
    global selected_point, running_idx, point_database, EDITOR_WINDOW
    real_x, real_y = EDITOR_WINDOW.convert_window_to_image_coords(x, y)


    location = bpy.context.scene.cursor.location
    location = np.array([location.x, location.y, location.z, 1.0])

    name = f"Reference_{running_idx}"
    running_idx += 1
    with db_lock:
        print(f"add point: {name} at image coords {real_x}, {real_y} -> world coords {location}")
        point_database[name] = [
            np.array([real_x, real_y]),
            location
        ]

    com = EditCommand()
    com.command = "ADD"
    com.location = location[:3]
    com.object_name = name
    COM_EDIT2BLENDER.put(com)
    redraw()

def select_click(x, y, flags, param):
    global selected_point, running_idx, point_database, EDITOR_WINDOW
    real_x, real_y = EDITOR_WINDOW.convert_window_to_image_coords(x, y)

    if(flags & cv2.EVENT_FLAG_SHIFTKEY) > 0:
        loc = np.array([real_x, real_y])
        with db_lock:
            for k in point_database:
                if np.linalg.norm(point_database[k][0] - loc) < 5:
                    selected_point = k

                    com = EditCommand()
                    com.command = "SEL"
                    com.object_name = selected_point

                    COM_EDIT2BLENDER.put(com)

        redraw()

def edit_thread():
    global TARGET_EDITABLE, EDITOR_OPEN, DATABASE_DIRTY, selected_point, point_database, EDITOR_WINDOW
    TARGET_EDITABLE = TARGET_FRAME.copy()

    print("thread started")
    redraw()
    EDITOR_WINDOW.add_mouse_callback(cv2.EVENT_LBUTTONDBLCLK, spawn_click)
    EDITOR_WINDOW.add_mouse_callback(cv2.EVENT_LBUTTONDOWN, select_click)

    EDITOR_OPEN = True
    try:
        
        while cv2.getWindowProperty(WINDOW_TITLE, 0) >= 0:
            a = cv2.waitKey(30)
            if DATABASE_DIRTY:
                selected_point = None
                redraw()
                DATABASE_DIRTY = False

            if a == 27:
                if selected_point is None:
                    break
                
                selected_point = None
                redraw()

            if a & 0xFF == ord('d'):
                if selected_point is None:
                    continue
                
                com = EditCommand()
                com.command = "DEL"
                com.object_name = selected_point
                
                COM_EDIT2BLENDER.put(com)
                with db_lock:
                    print(f"DEL: {selected_point}: {point_database[selected_point][0]} -> {point_database[selected_point][1]}")
                    del(point_database[selected_point])
                selected_point = None
                redraw()

    except Exception:
        print(traceback.format_exc())

    EDITOR_WINDOW.close()
    EDITOR_WINDOW = None

    print("Window Closed")

    with db_lock:
        for k in point_database:
            com = EditCommand()
            com.command = "DEL"
            com.object_name = k
            
            COM_EDIT2BLENDER.put(com)
            print(f"DEL: {k}: {point_database[k][0]} -> {point_database[k][1]}")
        point_database.clear()

    EDITOR_OPEN = False

    pass

def run_edit_thread():
    # Start render in a separate thread
    if not bpy.app.timers.is_registered(recieve_commands):
        bpy.app.timers.register(recieve_commands)

    thread = threading.Thread(target=edit_thread)
    thread.start()

##############################################
############### Commmunication UI ############
##############################################

def update_3d_positions():
    global point_database
    with db_lock:
        for k in point_database:
            obj = bpy.data.objects[k]
            loc = obj.location
            old = point_database[k][1]
            new = np.array([loc.x, loc.y, loc.z, 1.0])
            point_database[k][1] = new
            print(f"Update: {old} -> {new}")

def delete_by_name(name):
    obj = bpy.context.scene.objects.get(name)
    if obj:
        objs = bpy.data.objects
        objs.remove(obj, do_unlink=True)
    

# Function to spawn an empty at a given 3D position
def spawn_empty(location, name):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=location)
    newObj = bpy.data.objects['Empty']
    newObj.name = name

def spawn_empty_at_cursor(name):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=bpy.context.scene.cursor.location)
    newObj = bpy.data.objects['Empty']
    newObj.name = name

def select_reference(name):
    bpy.context.active_object.select_set(False)
    obj = bpy.data.objects[name]
    obj.select_set(True)

def recieve_commands():

    while not COM_EDIT2BLENDER.empty():
        com: EditCommand = COM_EDIT2BLENDER.get()
        if com.command == "DEL":
            delete_by_name(com.object_name)
        if com.command == "ADD":
            spawn_empty(com.location, com.object_name)
        if com.command == "ADC":
            spawn_empty_at_cursor(com.object_name)
        if com.command == "SEL":
            select_reference(com.object_name)

    return 0.5  # Check again in 0.5 seconds

##############################################
############### Blender UI ###################
##############################################

def get_camera_parameters():
    cam = bpy.data.objects['Camera']
    loc = cam.location
    rot = cam.rotation_euler
    f = cam.data.fisheye_lens
    s = cam.data.sensor_width

    info_dict = {
        "info":{
            "loc": [loc.x, loc.y, loc.z],
            "rot": [rot.x, rot.y, rot.z],
            "rot_info": rot.order,
            "f": f,
            "s": s
        },
        "points": {}
    }

    cinf = LoadSampleFromDict(info_dict)

    return np.array([ cinf.loc_inv[0],  cinf.loc_inv[1],  cinf.loc_inv[2], -cinf.rot_quat[0], cinf.rot_quat[1], cinf.rot_quat[2], cinf.rot_quat[3], cinf.alpha, 0, 0])


def create_camera_from_reference():
    global point_database

    # just for safety, update the 3d positions first
    update_3d_positions()

    cam = bpy.data.objects['Camera_Prior']
    loc = cam.location
    rot = cam.rotation_euler
    f = cam.data.fisheye_lens
    s = cam.data.sensor_width


    ########## use old way to optimize 
    params_old = optimize_camera_parameters([20, loc.x, loc.y, loc.z, math.degrees(rot.x), math.degrees(rot.y), math.degrees(rot.z), s])
    info_dict_debug = {
        "info":{
            "loc": params_old[1:4],
            "rot": [math.radians(params_old[4]), math.radians(params_old[5]), math.radians(params_old[6])],
            "rot_info": rot.order,
            "f": params_old[0],
            "s": s
        },
        "points": {}
    }
    cinf_debug = LoadSampleFromDict(info_dict_debug)
    params_debug = np.array([ cinf_debug.loc_inv[0],  cinf_debug.loc_inv[1],  cinf_debug.loc_inv[2], -cinf_debug.rot_quat[0], cinf_debug.rot_quat[1], cinf_debug.rot_quat[2], cinf_debug.rot_quat[3], cinf_debug.alpha, 0, 0])
    ##################################



    info_dict = {
        "info":{
            "loc": [loc.x, loc.y, loc.z],
            "rot": [rot.x, rot.y, rot.z],
            "rot_info": rot.order,
            "f": f,
            "s": s
        },
        "points": {}
    }

    cinf = LoadSampleFromDict(info_dict)
    # gt_params = get_camera_parameters()
    input_params = np.array([ cinf.loc_inv[0],  cinf.loc_inv[1],  cinf.loc_inv[2], -cinf.rot_quat[0], cinf.rot_quat[1], cinf.rot_quat[2], cinf.rot_quat[3], cinf.alpha, 0, 0])
    print("Point Database:")
    print(point_database)
    params, error, _ = RunCameraOptimization(initial_guess=input_params, point_database=point_database)
    print(f"Optimization finished with error: {error}")

    # print(f"Original Parameters: {gt_params}")
    print(f"Input Parameters: {input_params}")
    print(f"Estimated Parameters: {params}")
    print(f"DEBUG Parameters: {params_debug}")

    scn = bpy.context.scene
    # create camera
    cam1 = bpy.data.cameras.new("Estimated Camera")
    cam1.type = "PANO"
    cam1.panorama_type = "FISHEYE_EQUISOLID"

    alpha = params[7]
    cam1.fisheye_lens = cam1.sensor_width / (2 * alpha)

    alpha_inferred = cam1.sensor_width / (2 * cam1.fisheye_lens)
    print(f"Initial Alpha: {cinf.alpha}")
    print(f"Alpha: {alpha} -> {alpha_inferred}")

    cam_obj1 = bpy.data.objects.new("Estimated Camera", cam1)

    q_candidate = params[3:7]
    quat = q_candidate / np.linalg.norm(q_candidate)

    rotation = quaternion_to_euler(quaternion_conjugate(quat))
    inv_rotation = quaternion_to_euler(quat)
    location = InverseLocation(params[:3], inv_rotation)
    cam_obj1.location = (location[0], location[1], location[2])

    cam_obj1.rotation_euler = (math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))
    scn.collection.objects.link(cam_obj1)

    ######################################
    ######################################
    ######################################


    # render the estimated camera
    # print("Rendering Estimated Camera")
    # render_path = setup_renderer("color_estimated.png", bpy.data.objects["Estimated Camera"])
    # renderLayers, viewerNode = constructRenderGraph()
    # estimated_depth = get_pass_img("Position", 3, np.float32, renderLayers, viewerNode)
    # estimated_render = cv2.imread(str(render_path))

    # coord_world = []
    # coord_projected = []
    # center = np.array([1920 / 2, 1080 / 2])
    # offset = np.array([0.5, 0.5])
    # for k in point_database:
    #     point = point_database[k]
    #     coord_projected.append(point[0] - center + offset)
    #     coord_world.append(point[1][:3])

    # # batch the inputs together for easier compute
    # coord_world = np.array(coord_world)
    # coord_projected = np.array(coord_projected)
    # error = equisolid_linear_obj(params, coord_world, coord_projected, (1920, 1080), True, TARGET_FRAME, estimated_render, estimated_depth)

    ######################################
    ######################################
    ######################################

# Function to open the rendered image in a popup
def open_render_result():
    run_edit_thread()

def save_state():
    global point_database

    update_3d_positions()

    save_folder = Path(__file__).parent / "storage"
    save_folder.mkdir(exist_ok=True)
    i = 0

    while save_folder.exists():
        project_name = f'{i:06d}'
        save_folder = Path(__file__).parent / "storage" / project_name
        i += 1

    save_dict = {
        "points": {}
    }
    
    save_folder.mkdir(exist_ok=True)
    cam = bpy.data.objects['Camera']
    if not cam is None:
        loc = cam.location
        rot = cam.rotation_euler
        f = cam.data.fisheye_lens
        s = cam.data.sensor_width

        save_dict["info"]={
            "loc": [loc.x, loc.y, loc.z],
            "rot": [rot.x, rot.y, rot.z],
            "rot_info": rot.order,
            "f": f,
            "s": s
        }

    with db_lock:
        for k in point_database:
            save_dict["points"][k] = [point_database[k][0].tolist(), point_database[k][1].tolist()]
    with open(save_folder / "points.json", "w") as f:
        json.dump(save_dict, f)

    np.save(save_folder / "position", CURRENT_POSITION_BUFFER)
    np.save(save_folder / "color", CURRENT_RENDER)

    np.save(save_folder / "target", TARGET_FRAME)
    np.save(save_folder / "target_position", TARGET_POSITIONS_DEBUG)
    pass

def load_state(save_folder):
    global point_database, CURRENT_POSITION_BUFFER, CURRENT_RENDER, DATABASE_DIRTY, running_idx, TARGET_FRAME, TARGET_POSITIONS_DEBUG
    
    if not (save_folder / "position.npy").exists():
        return
    
    with db_lock:
        for k in point_database:
            delete_by_name(k)

    with open(save_folder / "points.json", "r") as f:
        load_dict_full = json.load(f)


    if "info" in load_dict_full:
        cam = bpy.data.objects['Camera']
        cam.location.x, cam.location.y, cam.location.z = load_dict_full["info"]["loc"]
        cam.rotation_euler.x, cam.rotation_euler.y, cam.rotation_euler.z = load_dict_full["info"]["rot"]

        cam.data.fisheye_lens = load_dict_full["info"]["f"]
        cam.data.sensor_width = load_dict_full["info"]["s"]

    load_dict = load_dict_full["points"]
    with db_lock:
        point_database.clear()
        running_idx = 0
        for k in load_dict:
            num = int(k.split("_")[1])
            running_idx = max(running_idx, num+1)
            point_database[k] = [
                np.array(load_dict[k][0]),
                np.array(load_dict[k][1])
            ]
            spawn_empty(point_database[k][1][:3], k)

    with open(save_folder / "position.npy", "rb") as f:
        CURRENT_POSITION_BUFFER = np.load(f)

    with open(save_folder / "color.npy", "rb") as f:
        CURRENT_RENDER = np.load(f)

    with open(save_folder / "target.npy", "rb") as f:
        TARGET_FRAME = np.load(f)

    with open(save_folder / "target_position.npy", "rb") as f:
        TARGET_POSITIONS_DEBUG = np.load(f)

    if not EDITOR_OPEN:
        open_render_result()

    DATABASE_DIRTY = True


class OptimizationPanel(bpy.types.Panel):
    """ Creates a panel in the UI to trigger the render """
    bl_label = "Camera Optimization"
    bl_idname = "RENDER_PT_progress"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CamAlign"

    def draw(self, context):
        layout = self.layout
        layout.operator("opticam.start_render")
        
        layout.operator("opticam.open_filebrowser")
        layout.operator("opticam.open_render")
        
        layout.operator("opticam.find_matches")
        
        # layout.operator("opticam.update_refs")
        layout.operator("opticam.create_camera")

class ProjectsPanel(bpy.types.Panel):
    """ Creates a panel in the UI to trigger the render """
    bl_label = "Projects"
    bl_idname = "RENDER_PT_projects"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "CamAlign"

    def draw(self, context):
        layout = self.layout
        
        layout.operator("opticam.load_state")
        layout.operator("opticam.save_state")

class StartRenderOperator(bpy.types.Operator):
    """ Operator to start the render """
    bl_idname = "opticam.start_render"
    bl_label = "Run Setup"

    def invoke(self, context, event):
        run_render()
        return {'FINISHED'}
    
class FindMatchesOperator(bpy.types.Operator):
    bl_idname = "opticam.find_matches"
    bl_label = "Automatically Find Matches"

    @classmethod
    def poll(cls, context):
        global TARGET_FRAME, CURRENT_RENDER
        return TARGET_FRAME is not None and CURRENT_RENDER is not None
    
    def invoke(self, context, event):
        populatePointDatabase()
        return {'FINISHED'}


# Operator to open the render result and activate click handler
class IMAGE_OT_open_render(bpy.types.Operator):
    bl_idname = "opticam.open_render"
    bl_label = "Open Editor"


    @classmethod
    def poll(cls, context):
        global TARGET_FRAME
        return TARGET_FRAME is not None
    
    def execute(self, context):
        # Open the rendered image in an image editor
        open_render_result()
        return {'FINISHED'}
    
class UpdatePointsOperator(bpy.types.Operator):
    bl_idname = "opticam.update_refs"
    bl_label = "Update References"
    
    def execute(self, context):
        # Open the rendered image in an image editor
        update_3d_positions()
        return {'FINISHED'}
    
class CreateCameraOperator(bpy.types.Operator):
    bl_idname = "opticam.create_camera"
    bl_label = "Run Optimizatrion"
    
    def execute(self, context):
        # Open the rendered image in an image editor
        create_camera_from_reference()
        return {'FINISHED'}
    
class SaveStateOperator(bpy.types.Operator):
    bl_idname = "opticam.save_state"
    bl_label = "Save State"
    
    def execute(self, context):
        # Open the rendered image in an image editor
        save_state()
        return {'FINISHED'}
    
class LoadStateOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "opticam.load_state"
    bl_label = "Load State"
    filepath = Path(__file__).parent / "addon_data"
    
    def execute(self, context):
        # Print or use the selected file path
        print(f"Selected file: {self.filepath}")
        p = Path(self.filepath).parent
        load_state(p)
        return {'FINISHED'}
    
    def invoke(self, context, event):
        # Set the default directory here
        self.filepath = str(Path(__file__).parent / "addon_data")
        return super().invoke(context, event)

    
class OpenFileBrowserOperator(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    bl_idname = "opticam.open_filebrowser"
    bl_label = "Load Camera Frame"

    filepath = "."

    def execute(self, context):
        global TARGET_FRAME
        # Print or use the selected file path
        print(f"Selected file: {self.filepath}")
        TARGET_FRAME = cv2.imread(str(self.filepath))
        return {'FINISHED'}
    
classes = [
    OptimizationPanel,
    ProjectsPanel,
    StartRenderOperator,
    UpdatePointsOperator,
    CreateCameraOperator,
    SaveStateOperator,
    LoadStateOperator,
    IMAGE_OT_open_render,
    FindMatchesOperator,
    OpenFileBrowserOperator
]

def register():
    print("Registering CamAlign Addon")
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
