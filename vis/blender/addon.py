import bpy # pyright: ignore
import os
from mathutils import Quaternion, Vector # pyright: ignore
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from vis.blender.utils import SMPLX_JOINT_NAMES


def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result
    return
        

def add_smplx_model(gender='neutral', name='SMPLX_Character', model_dir='./deps/blender', use_300_shape=True):
    """
    Add an SMPL-X model of a given gender to the Blender scene with a custom name.

    Parameters:
    - gender (str): 'male', 'female', or 'neutral'
    - name (str): Unique name to assign to the character
    - model_dir (str): Absolute path to the folder containing the SMPL-X .blend files
    - use_300_shape (bool): Whether to use the 300-shape blend file (if available)
    """
    assert gender in ['male', 'female', 'neutral'], "Gender must be 'male', 'female' or 'neutral'."

    model_file_300 = "smplx_model_300_20220615.blend"
    model_file_default = "smplx_model_20210421.blend"
    
    model_file = model_file_300 if use_300_shape else model_file_default
    model_path = os.path.join(model_dir, model_file)
    objects_path = os.path.join(model_path, "Object")

    object_name = f"SMPLX-mesh-{gender}"
    armature_name = f"SMPLX-{gender}"

    # Append mesh
    bpy.ops.wm.append(filename=object_name, directory=objects_path)

    # Rename and reselect
    imported_mesh = bpy.context.selected_objects[0]
    imported_armature = None

    # Identify the armature (the other selected object)
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            imported_armature = obj

    if imported_mesh is None or imported_armature is None:
        raise RuntimeError("Failed to import SMPL-X model.")

    # Rename
    imported_mesh.name = f"{name}_mesh"
    imported_armature.name = f"{name}_armature"

    # Parent the mesh to the armature
    imported_mesh.select_set(True)
    imported_armature.select_set(True)
    bpy.context.view_layer.objects.active = imported_armature
    bpy.ops.object.parent_set(type='ARMATURE')


    return imported_armature, imported_mesh


def add_smplx_model_sequence(gender='neutral', name='SMPLX_Character', number=5,
                             model_dir='./deps/blender', use_300_shape=True):
    """
    Add a sequence of SMPL-X models of a given gender to the Blender scene with a custom name.
    """
    armatures = []
    meshes = []
    
    for i in range(number):
        a, m = add_smplx_model(gender=gender, name=f'{name}_{i}', model_dir=model_dir, use_300_shape=use_300_shape)
        armatures.append(a)
        meshes.append(m)
    
    return armatures, meshes
    


def get_bone_world_head_z(armature_obj, bone_name):
    # Get the world matrix of the pose bone's head
    bone = armature_obj.pose.bones.get(bone_name)
    if not bone:
        raise ValueError(f"Bone '{bone_name}' not found")

    bone_matrix = armature_obj.matrix_world @ bone.matrix
    head_world = bone_matrix.translation
    return head_world.z

def get_lowest_foot_z(armature_obj, foot_bones, start_frame, end_frame):
    min_z = float('inf')

    for f in range(start_frame, end_frame + 1):
        bpy.context.scene.frame_set(f)
        for bone_name in foot_bones:
            z = get_bone_world_head_z(armature_obj, bone_name)
            if z < min_z:
                min_z = z

    return min_z

def lift_armature_to_ground_from_feet(armature_obj, frames=[0,1]):
    foot_bones = ['left_foot', 'right_foot']

    start_frame = frames[0]
    end_frame = frames[1]

    min_z = get_lowest_foot_z(armature_obj, foot_bones, start_frame, end_frame)
    offset = -min_z
    print(f"Lowest foot Z over frames {start_frame}-{end_frame}: {min_z:.4f}")
    print(f"Applying offset: {offset:.4f}")
    print('-------------------------------------------------')

    armature_obj.location.z += offset
    armature_obj.keyframe_insert(data_path="location", frame=start_frame)
    
    return offset




def add_smplx_animations(armature, trans:np.ndarray, poses:np.ndarray):
    """ armature: bpy.object.armature.
        trans: np.ndarray, [N, 3].
        poses: np.ndarray, [N, 165].
    """
    assert trans.shape[0] == poses.shape[0], "Invalid motion shape."
    nj = len(SMPLX_JOINT_NAMES) 
    poses = poses.reshape(-1, nj, 3) # -1 for root nj is 56 but num j is 55
    for i in range(poses.shape[0]):
        for j in range(nj):
            bone = armature.pose.bones[SMPLX_JOINT_NAMES[j]]
            bone.rotation_mode = 'QUATERNION'
            rod = Vector(poses[i, j, :])
            angle_rad = rod.length
            axis = rod.normalized()
            quat = Quaternion(axis, angle_rad)
            bone.rotation_quaternion = quat
            bone.keyframe_insert(data_path="rotation_quaternion", frame=i+1)
            
        armature.pose.bones['root'].location = trans[i, :]
        armature.pose.bones['root'].keyframe_insert(data_path="location", frame=i+1)
    

def add_smplx_pose_seqeunce(armatures:list, trans:np.ndarray, poses:np.ndarray):
    """
    This is used to add a sequence of smplx in a single frame all together
    Args:
        armatures: list[bpy.types.Armature].
        trans: np.ndarray, [N, 3].
        poses: np.ndarray, [N, 165].
    """
    assert len(armatures) == len(trans) == len(poses), "Invalid motion shape."
    
    for i in range(len(armatures)):
        add_smplx_animations(armatures[i], trans[i:i+1], poses[i:i+1])
    return


from utils.constant import relaxed_hand_pose



def collate_motions(motions:np.ndarray):
    """
    Args:
        motions (np.ndarray): [N, 55 / 52 / 22, 3].

    Returns:
        np.ndarray: [N, 55, 3].
    """
    assert motions.shape[1] in [55, 52, 22], "Invalid motion shape."
    
    result = np.zeros((motions.shape[0], 55, 3))
    
    if motions.shape[1] == 55:
        return motions
    elif motions.shape[1] == 52: # no yaw eys
        result[:, :15, :] = motions[:, :15, :]
        result[:, 15 + 3:, :] = motions[:, 15:, :]
    elif motions.shape[1] == 22:
        result[:, :15, :] = motions[:, :15, :]
        result[:, 18:22, :] = motions[:, 15:27, :]
        result[:, -(30 * 3):, :] = relaxed_hand_pose
    else:
        raise ValueError("Invalid motion shape.")


def remove_cube():
    for obj in bpy.data.objects:
        if obj.name == 'Cube':
            bpy.data.objects.remove(obj)
    return

def set_white_background():
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    render_layers = nodes.new(type='CompositorNodeRLayers')
    alpha_over = nodes.new(type='CompositorNodeAlphaOver')
    white = nodes.new(type='CompositorNodeRGB')
    white.outputs[0].default_value = (1, 1, 1, 1)
    composite = nodes.new(type='CompositorNodeComposite')

    # Link nodes
    links.new(render_layers.outputs['Image'], alpha_over.inputs[2])  # Foreground
    links.new(white.outputs[0], alpha_over.inputs[1])                # Background
    links.new(alpha_over.outputs[0], composite.inputs[0])

def set_white_background_old():
    bpy.context.scene.world.use_nodes = True
    world = bpy.context.scene.world
    node_tree = world.node_tree
    for node in node_tree.nodes:
            node_tree.nodes.remove(node)

    # Add Background node
    background_node = node_tree.nodes.new(type='ShaderNodeBackground')
    background_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # White color for ambient light
    background_node.inputs['Strength'].default_value = 1.0  # Adjust the strength of the ambient light
    bpy.context.scene.view_settings.view_transform = 'Standard'


    # Add World Output node
    world_output_node = node_tree.nodes.new(type='ShaderNodeOutputWorld')

    # Link nodes
    node_tree.links.new(background_node.outputs['Background'], world_output_node.inputs['Surface'])

    print("Ambient light added successfully.")


def hhi_dye_two(mesh1, 
                mesh2, 
                color1 = (0.8, 0.022, 0.014, 1), 
                color2 = (0.0, 0.417, 0.8, 1)):
    new_material1 = bpy.data.materials.new(name=f"p1_dye")
    new_material1.diffuse_color = color1
    new_material1.roughness = 1.0
    new_material1.specular_intensity=0.05
    if mesh1.data.materials:
        mesh1.data.materials[0] = new_material1
    else:
        mesh1.data.materials.append(new_material1)    

    new_material2 = bpy.data.materials.new(name=f"p2_dye")
    new_material2.diffuse_color = color2
    new_material2.roughness = 1.0
    new_material2.specular_intensity=0.05
    if mesh2.data.materials:
        mesh2.data.materials[0] = new_material2
    else:
        mesh2.data.materials.append(new_material2)    
    return

def lighten_color(rgba, ratio):
    '''ratio = 0, pure rgb, ratio = 1, pure white'''
    r, g, b, a = rgba
    r_new = ratio * (1 - r) + r
    g_new = ratio * (1 - g) + g
    b_new = ratio * (1 - b) + b
    return (r_new, g_new, b_new, a)

def hhi_dye_sequence(meshes1, 
                     meshes2, 
                     color1 = (0.8, 0.022, 0.014, 1), 
                     color2 = (0.0, 0.417, 0.8, 1)):
    ratio = [0.2 * (1- i / (len(meshes1)-1)) for i in range(len(meshes1))]
    
    print(ratio, '================')
    
    for i in range(len(meshes1)):
        color1_new = lighten_color(color1, ratio[i])
        color2_new = lighten_color(color2, ratio[i])
        hhi_dye_two(meshes1[i], meshes2[i], color1_new, color2_new)
    return

def set_area_light(size=(3, 15), power=200, location=(0, 0, 4)):

    # Create new area light data
    light_data = bpy.data.lights.new(name="SoftKey", type='AREA')
    light_data.energy = power
    light_data.shape = 'RECTANGLE'
    light_data.size = size[0]  # X dimension
    light_data.size_y = size[1]  # Z (vertical) dimension

    # Create new light object
    light_obj = bpy.data.objects.new(name="SoftKey", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    # Position it somewhere
    light_obj.location = (0, 0, 4)

    return light_obj


def set_light_power(power=300):
    bpy.data.objects['Light'].data.energy = power
    light = bpy.data.lights.new(name="SoftKey", type='AREA')
    light_obj = bpy.data.objects.new(name="SoftKey", object_data=light)
    bpy.context.collection.objects.link(light_obj)

    light_obj.location = (0, 0, 4)
    light.energy = 200  # Experiment here
    light.shape = 'SQUARE'
    light.size = 5.0
    return

def add_image_texture_to_plane(image_path, size=5):
    # Add a plane mesh
    bpy.ops.mesh.primitive_plane_add(size=size)  # Adjust the size as needed
    plane = bpy.context.object
    
    # 2. Load the Image
    # Check if the image is already loaded
    if image_path not in bpy.data.images:
        # Load image if not already loaded
        image = bpy.data.images.load(image_path)
    else:
        image = bpy.data.images[image_path]
    
    # 3. Create a Material
    material = bpy.data.materials.new(name="TextureMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    
    # Create a texture node
    texture_node = material.node_tree.nodes.new(type='ShaderNodeTexImage')
    texture_node.image = image
    
    # Connect the texture node to the BSDF node
    material.node_tree.links.new(bsdf.inputs['Base Color'], texture_node.outputs['Color'])
    
    # 4. Assign the Material to the Plane
    if len(plane.data.materials) > 0:
        plane.data.materials[0] = material
    else:
        plane.data.materials.append(material)
        
    plane.location = (0, 0, -0.01)


def add_floor(floor_color=(0.1, 0.1, 0.1, 1), size=5):
    # Always add plane with default size
    bpy.ops.mesh.primitive_plane_add(size=1)
    plane_obj = bpy.context.object

    # Scale the plane
    if isinstance(size, (list, tuple)) and len(size) == 2:
        plane_obj.scale[0] = size[0]  # X
        plane_obj.scale[1] = size[1]  # Y
    else:
        plane_obj.scale[0] = size
        plane_obj.scale[1] = size
    # Create and assign material
    mat = bpy.data.materials.new(name="GdataroundMaterial")
    mat.diffuse_color = floor_color
    if plane_obj.data.materials:
        plane_obj.data.materials[0] = mat
    else:
        plane_obj.data.materials.append(mat)

    return plane_obj

def render_mp4(filepath, frames):
    output_path = filepath
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'

    # Set output directory and filename
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    bpy.context.scene.render.filepath = output_path

    # Set start and end frames for the animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frames  # Adjust frame range as needed

    # Render animation
    bpy.ops.render.render(animation=True)

def get_max_keyframe_frame():
    max_frame = None

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                for kp in fcurve.keyframe_points:
                    frame = kp.co[0]
                    if max_frame is None or frame > max_frame:
                        max_frame = frame

    return int(max_frame) if max_frame is not None else None    

def render_mp4_blender():
    blend_file_path = bpy.data.filepath
    output_path = os.path.dirname(blend_file_path)

    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'

    # Set output directory and filename
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    bpy.context.scene.render.filepath = output_path

    # Set start and end frames for the animation
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = get_max_keyframe_frame()

    # Render animation
    bpy.ops.render.render(animation=True)


def render_current_frame(filepath):
    """ copy this to blender file and """
    import os
    import bpy # pyright: ignore

    bpy.context.scene.world.use_nodes = True
    world = bpy.context.scene.world
    node_tree = world.node_tree

    # Clear existing nodes
    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    output_path = filepath

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.resolution_x = 3840 * 2 # Set the horizontal resolution
    bpy.context.scene.render.resolution_y = 2160 * 2 # Set the vertical resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.filepath = output_path

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Set the frame and render
    bpy.context.scene.frame_set(1)
    bpy.ops.render.render(write_still=True)

    print("Render complete. Check the output file at:", output_path)
    

def delete_objects_by_name(*names):
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)

def full_delete_object(name):
    obj = bpy.data.objects.get(name)
    if obj:
        obj_type = obj.type
        obj_data = obj.data

        # Unlink from all collections
        for collection in obj.users_collection:
            collection.objects.unlink(obj)

        # Remove the object
        bpy.data.objects.remove(obj, do_unlink=True)

        # Then clean up the data if it's not used elsewhere
        if obj_type == 'MESH' and obj_data.name in bpy.data.meshes:
            bpy.data.meshes.remove(obj_data, do_unlink=True)
        elif obj_type == 'ARMATURE' and obj_data.name in bpy.data.armatures:
            bpy.data.armatures.remove(obj_data, do_unlink=True)


if __name__ == "__main__":
    remove_cube()
    set_white_background()
    set_light_power(power=600)
    add_floor()

    
    
    for i in range(10):
    
        a1, m1 = add_smplx_model(gender='neutral', name=f'p1_{i}')
        a2, m2 = add_smplx_model(gender='neutral', name=f'p2_{i}')

        motion_data1 = np.load('j2s1p1.npz', allow_pickle=True)
        motion_data2 = np.load('j2s1p2.npz', allow_pickle=True)

        add_smplx_animations(a1, motion_data1['trans'], motion_data1['poses'])
        add_smplx_animations(a2, motion_data2['trans'], motion_data2['poses'])

        hhi_dye_two(m1, m2)

        render_mp4(filepath=f"output/j2s_index{i}.mp4", frames=motion_data1['poses'].shape[0])
        bpy.ops.wm.save_as_mainfile(filepath=f"output/j2s_index{i}.blend")
        
        delete_objects_by_name(f'p1_{i}_armature', f'p2_{i}_armature')
        delete_objects_by_name(f'p1_{i}_mesh', f'p2_{i}_mesh')