import blenderproc as bproc
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
from PIL import Image

from blenderproc.python.utility.GlobalStorage import GlobalStorage
from blenderproc.python.writer.WriterUtility import _WriterUtility


TEXTURE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MODEL_SUFFIXES = {".obj", ".ply", ".stl", ".fbx", ".glb", ".gltf"}
HDRI_SUFFIXES = {".hdr", ".exr"}
BACKGROUND_CATEGORY_ID = 0
GENERATED_MATERIAL_PREFIXES = ("floor_mat_", "background_floor_mat_", "room_wall_mat_", "box_mat_", "part_mat_")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic stacked-part scenes with BlenderProc."
    )
    parser.add_argument("--parts-dir", default="assets/parts",
                        help="Directory containing CAD part models.")
    parser.add_argument("--image-dir", default="assets",
                        help="Directory containing backgrounds, HDRIs, and textures.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory in which the generated dataset is stored.")
    parser.add_argument("--dataset-name", default="synmech_dataset")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--start-image-id", type=int, default=None)
    parser.add_argument("--min-parts", type=int, default=5)
    parser.add_argument("--max-parts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--background-mode", choices=("backgrounds", "hdris"), default="hdris")
    parser.add_argument("--scene-style-dir", default="")
    parser.add_argument("--ground-texture-dir", default="")
    parser.add_argument("--container-texture-dir", default="")
    parser.add_argument("--part-texture-dir", default="")
    parser.add_argument("--disable-surface-wear", action="store_true")

    parser.add_argument("--hdri-dir", default="")
    parser.add_argument("--hdri-strength-min", type=float, default=0.5)
    parser.add_argument("--hdri-strength-max", type=float, default=1.8)
    parser.add_argument("--disable-walls", action="store_true")

    parser.add_argument("--floor-size-x", type=float, default=0.0001)
    parser.add_argument("--floor-size-y", type=float, default=0.0001)
    parser.add_argument("--floor-thickness", type=float, default=0.05)
    parser.add_argument("--background-floor-gap", type=float, default=0.01)

    parser.add_argument("--bin-length", type=float, default=0.265)
    parser.add_argument("--bin-width", type=float, default=0.165)
    parser.add_argument("--bin-height", type=float, default=0.115)
    parser.add_argument("--wall-thickness", type=float, default=0.01)

    parser.add_argument("--drop-height-min", type=float, default=0.20)
    parser.add_argument("--drop-height-max", type=float, default=1.20)
    parser.add_argument("--drop-xy-margin", type=float, default=0.05)
    parser.add_argument("--initial-z-spacing", type=float, default=0.025)
    parser.add_argument("--init-overlap-margin", type=float, default=0.003)
    parser.add_argument("--max-initial-pose-tries", type=int, default=500)

    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--collision-shape", default="CONVEX_HULL",
                        choices=["BOX", "SPHERE", "CAPSULE", "CYLINDER", "CONE", "CONVEX_HULL", "MESH"])
    parser.add_argument("--gravity-scale", type=float, default=1.0)
    parser.add_argument("--linear-damping", type=float, default=0.35)
    parser.add_argument("--angular-damping", type=float, default=0.50)
    parser.add_argument("--physics-min-time", type=float, default=2.5)
    parser.add_argument("--physics-max-time", type=float, default=7.5)
    parser.add_argument("--physics-substeps", type=int, default=35)
    parser.add_argument("--physics-solver-iters", type=int, default=40)
    parser.add_argument("--max-scene-tries", type=int, default=8)

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fx", type=float, default=1280.0)
    parser.add_argument("--fy", type=float, default=1280.0)
    parser.add_argument("--cx", type=float, default=-1.0)
    parser.add_argument("--cy", type=float, default=-1.0)
    parser.add_argument("--camera-radius-min", type=float, default=0.30)
    parser.add_argument("--camera-radius-max", type=float, default=0.35)
    parser.add_argument("--camera-elev-min", type=float, default=89.0)
    parser.add_argument("--camera-elev-max", type=float, default=89.0)
    parser.add_argument("--camera-inplane-deg", type=float, default=8.0)

    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--color-format", default="PNG", choices=["PNG", "JPEG"])
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--frames-per-chunk", type=int, default=6000)
    parser.add_argument("--bop-num-worker", type=int, default=0)
    parser.add_argument("--calc-bop-masks", action="store_true")
    parser.add_argument("--no-bop-masks", action="store_true")
    parser.add_argument("--min-visible-pixels", type=int, default=120)
    parser.add_argument("--min-visible-instances", type=int, default=1)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=2500)
    return parser.parse_args()

def set_seed(seed: int) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def validate_args(args: argparse.Namespace) -> None:
    """在创建输出目录和运行渲染之前检查会导致无效数据的参数。"""
    positive = (
        "num_images", "frames_per_chunk", "samples", "width", "height", "fx", "fy", "depth_scale",
        "floor_size_x", "floor_size_y", "floor_thickness", "bin_length", "bin_width", "bin_height",
        "wall_thickness", "model_scale", "gravity_scale", "physics_min_time", "physics_max_time",
        "physics_substeps", "physics_solver_iters", "max_scene_tries", "max_initial_pose_tries",
        "camera_radius_min", "camera_radius_max", "min_parts", "max_parts", "min_visible_instances",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name} must be positive and finite")
    nonnegative = ("drop_height_min", "drop_xy_margin", "initial_z_spacing", "init_overlap_margin",
                   "camera_inplane_deg", "min_visible_pixels", "min_valid_depth_pixels", "bop_num_worker",
                   "hdri_strength_min", "hdri_strength_max")
    for name in nonnegative:
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be non-negative and finite")
    for low, high in (("min_parts", "max_parts"), ("camera_radius_min", "camera_radius_max"),
                      ("camera_elev_min", "camera_elev_max"), ("hdri_strength_min", "hdri_strength_max")):
        if not math.isfinite(getattr(args, high)) or getattr(args, low) > getattr(args, high):
            raise ValueError(f"--{low} must be <= --{high}, both finite")
    if not 0 <= args.camera_elev_min <= args.camera_elev_max <= 90:
        raise ValueError("Camera elevation must satisfy 0 <= min <= max <= 90 degrees")
    if not math.isfinite(args.drop_height_max) or args.drop_height_min >= args.drop_height_max:
        raise ValueError("--drop_height_min must be smaller than --drop_height_max")
    if args.physics_min_time >= args.physics_max_time:
        raise ValueError("--physics_min_time must be smaller than --physics_max_time")
    for name in ("linear_damping", "angular_damping"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name} must be between 0 and 1")
    if args.start_image_id is not None and args.start_image_id < 0:
        raise ValueError("--start-image-id must be a non-negative integer")
    if args.min_visible_instances > args.max_parts:
        raise ValueError("--min_visible_instances cannot exceed --max_parts")
    if max(args.min_visible_pixels, args.min_valid_depth_pixels) > args.width * args.height:
        raise ValueError("Pixel quality thresholds cannot exceed the image pixel count")
    if not 0 <= args.jpg_quality <= 100:
        raise ValueError("--jpg_quality must be between 0 and 100")
    if not math.isfinite(args.cx) or not math.isfinite(args.cy):
        raise ValueError("Camera principal point must be finite")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_files(folder: Path, suffixes: set) -> List[Path]:
    if not folder or not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def resolve_dir(image_dir: Path, explicit_dir: str, default_subdir: str) -> Path:
    if explicit_dir:
        return Path(explicit_dir).resolve()
    return (image_dir / default_subdir).resolve()


def semantic_name_from_stem(stem: str) -> str:
    """Create a stable semantic label from a CAD file name."""
    label = stem.strip().lower().replace("-", "_").replace(" ", "_")
    return label or "part"


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def first_frame(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def set_cp(obj: bproc.types.MeshObject, key: str, value: Any) -> None:
    obj.set_cp(key, value)


def assign_pass_indices() -> None:
    for index, obj in enumerate(bproc.object.get_all_mesh_objects(), start=1):
        obj.blender_obj.pass_index = index


def set_shader_value(material: Any, names: Sequence[str], value: Any) -> bool:
    last_error = None
    for name in names:
        try:
            material.set_principled_shader_value(name, value)
            return True
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not set material input {list(names)}") from last_error


def set_node_input_value(node: Any, names: Sequence[str], value: Any) -> bool:
    for name in names:
        try:
            if name in node.inputs:
                node.inputs[name].default_value = value
                return True
        except Exception:
            continue
    return False


def get_socket(sockets: Any, names: Sequence[str]) -> Any:
    for name in names:
        try:
            if name in sockets:
                return sockets[name]
        except Exception:
            continue
    return None


def remove_input_links(material: Any, input_socket: Any) -> None:
    while input_socket is not None and input_socket.links:
        link = input_socket.links[0]
        try:
            material.links.remove(link)
        except Exception:
            material.blender_obj.node_tree.links.remove(link)


def random_neutral_metal_color() -> List[float]:
    base = float(np.random.uniform(0.42, 0.82))
    tint = np.random.uniform(-0.035, 0.035, size=3)
    color = np.clip(base + tint, 0.25, 0.95)
    return [float(value) for value in color] + [1.0]


def add_brushed_metal_detail(material: Any) -> Dict[str, Any]:
    detail = {
        "profile": "brushed_random_metal",
        "brushed_metal": True,
        "anisotropic": float(np.random.uniform(0.35, 0.85)),
        "anisotropic_rotation": float(np.random.uniform(0.0, 1.0)),
        "line_scale": float(np.random.uniform(55.0, 180.0)),
        "line_distortion": float(np.random.uniform(2.0, 12.0)),
        "bump_strength": float(np.random.uniform(0.025, 0.085)),
        "bump_distance": float(np.random.uniform(0.00025, 0.00125)),
        "line_direction": random.choice(["X", "Y", "Z"]),
    }

    try:
        principled = material.get_the_one_node_with_type("BsdfPrincipled")
    except Exception:
        for key, value in detail.items():
            material.blender_obj[key] = value
        return detail

    set_shader_value(material, ["Anisotropic", "Anisotropic IOR Level"], detail["anisotropic"])
    set_shader_value(material, ["Anisotropic Rotation"], detail["anisotropic_rotation"])

    texcoord = material.new_node("ShaderNodeTexCoord", add_brushed_metal_detail.__name__)
    texcoord.location = (-900, -260)

    mapping = material.new_node("ShaderNodeMapping", add_brushed_metal_detail.__name__)
    mapping.location = (-700, -260)

    wave = material.new_node("ShaderNodeTexWave", add_brushed_metal_detail.__name__)
    wave.location = (-500, -260)
    try:
        wave.wave_type = "BANDS"
    except Exception:
        pass
    try:
        wave.bands_direction = detail["line_direction"]
    except Exception:
        pass
    set_node_input_value(wave, ["Scale"], detail["line_scale"])
    set_node_input_value(wave, ["Distortion"], detail["line_distortion"])

    colorramp = material.new_node("ShaderNodeValToRGB", add_brushed_metal_detail.__name__)
    colorramp.location = (-300, -260)
    dark_pos = float(np.random.uniform(0.38, 0.48))
    light_pos = float(np.random.uniform(0.52, 0.64))
    colorramp.color_ramp.elements[0].position = dark_pos
    colorramp.color_ramp.elements[1].position = light_pos
    detail["ramp_dark_position"] = dark_pos
    detail["ramp_light_position"] = light_pos

    bump = material.new_node("ShaderNodeBump", add_brushed_metal_detail.__name__)
    bump.location = (-80, -300)
    set_node_input_value(bump, ["Strength"], detail["bump_strength"])
    set_node_input_value(bump, ["Distance"], detail["bump_distance"])

    try:
        material.link(texcoord.outputs["Object"], mapping.inputs["Vector"])
        material.link(mapping.outputs["Vector"], wave.inputs["Vector"])
        material.link(wave.outputs["Fac"], colorramp.inputs["Fac"])
        material.link(colorramp.outputs["Color"], bump.inputs["Height"])
        if "Normal" in principled.inputs:
            remove_input_links(material, principled.inputs["Normal"])
            material.link(bump.outputs["Normal"], principled.inputs["Normal"])
    except Exception as exc:
        material.blender_obj["brushed_detail_warning"] = f"{type(exc).__name__}: {exc}"

    for key, value in detail.items():
        material.blender_obj[key] = value
    return detail


def sample_surface_wear_profile(base_roughness: float) -> Dict[str, Any]:
    profile = random.choices(
        ["clean", "light_wear", "oily_smudged", "heavy_wear"],
        weights=[0.25, 0.40, 0.25, 0.10],
        k=1,
    )[0]

    profile_ranges = {
        "clean": {
            "roughness_noise_strength": (0.015, 0.045),
            "scratch_bump_strength": (0.001, 0.006),
            "scratch_bump_distance": (0.000010, 0.000045),
            "dirt_strength": (0.0, 0.015),
            "oil_strength": (0.0, 0.015),
        },
        "light_wear": {
            "roughness_noise_strength": (0.045, 0.105),
            "scratch_bump_strength": (0.006, 0.018),
            "scratch_bump_distance": (0.000030, 0.000120),
            "dirt_strength": (0.015, 0.060),
            "oil_strength": (0.000, 0.050),
        },
        "oily_smudged": {
            "roughness_noise_strength": (0.080, 0.160),
            "scratch_bump_strength": (0.004, 0.014),
            "scratch_bump_distance": (0.000020, 0.000090),
            "dirt_strength": (0.025, 0.090),
            "oil_strength": (0.080, 0.220),
        },
        "heavy_wear": {
            "roughness_noise_strength": (0.120, 0.240),
            "scratch_bump_strength": (0.014, 0.032),
            "scratch_bump_distance": (0.000070, 0.000220),
            "dirt_strength": (0.080, 0.180),
            "oil_strength": (0.000, 0.080),
        },
    }
    ranges = profile_ranges[profile]
    detail = {
        "surface_profile": profile,
        "roughness_noise_scale": float(np.random.uniform(8.0, 38.0)),
        "roughness_noise_detail": float(np.random.uniform(4.0, 13.0)),
        "roughness_noise_roughness": float(np.random.uniform(0.48, 0.72)),
        "scratch_scale": float(np.random.uniform(180.0, 760.0)),
        "scratch_distortion": float(np.random.uniform(5.0, 28.0)),
        "scratch_direction": random.choice(["X", "Y", "Z"]),
    }

    for key, value_range in ranges.items():
        detail[key] = float(np.random.uniform(*value_range))

    roughness_delta = detail["roughness_noise_strength"]
    roughness_min = base_roughness - roughness_delta * 0.45 - detail["oil_strength"] * 0.35
    roughness_max = base_roughness + roughness_delta + detail["dirt_strength"] * 0.45
    roughness_min = float(np.clip(roughness_min, 0.035, 0.95))
    roughness_max = float(np.clip(roughness_max, roughness_min + 0.015, 1.0))
    detail["roughness_map_min"] = roughness_min
    detail["roughness_map_max"] = roughness_max
    detail["base_roughness_before_surface_wear"] = float(base_roughness)
    return detail


def add_surface_wear_detail(material: Any, base_roughness: float) -> Dict[str, Any]:
    detail = sample_surface_wear_profile(base_roughness)

    try:
        principled = material.get_the_one_node_with_type("BsdfPrincipled")
    except Exception:
        for key, value in detail.items():
            material.blender_obj[key] = value
        return detail

    texcoord = material.new_node("ShaderNodeTexCoord", add_surface_wear_detail.__name__)
    texcoord.location = (-1100, 40)

    roughness_noise = material.new_node("ShaderNodeTexNoise", add_surface_wear_detail.__name__)
    roughness_noise.location = (-900, 40)
    set_node_input_value(roughness_noise, ["Scale"], detail["roughness_noise_scale"])
    set_node_input_value(roughness_noise, ["Detail"], detail["roughness_noise_detail"])
    set_node_input_value(roughness_noise, ["Roughness"], detail["roughness_noise_roughness"])

    roughness_map = material.new_node("ShaderNodeMapRange", add_surface_wear_detail.__name__)
    roughness_map.location = (-690, 40)
    set_node_input_value(roughness_map, ["From Min"], 0.0)
    set_node_input_value(roughness_map, ["From Max"], 1.0)
    set_node_input_value(roughness_map, ["To Min"], detail["roughness_map_min"])
    set_node_input_value(roughness_map, ["To Max"], detail["roughness_map_max"])
    try:
        roughness_map.clamp = True
    except Exception:
        pass

    scratch_wave = material.new_node("ShaderNodeTexWave", add_surface_wear_detail.__name__)
    scratch_wave.location = (-900, -520)
    try:
        scratch_wave.wave_type = "BANDS"
    except Exception:
        pass
    try:
        scratch_wave.bands_direction = detail["scratch_direction"]
    except Exception:
        pass
    set_node_input_value(scratch_wave, ["Scale"], detail["scratch_scale"])
    set_node_input_value(scratch_wave, ["Distortion"], detail["scratch_distortion"])

    scratch_ramp = material.new_node("ShaderNodeValToRGB", add_surface_wear_detail.__name__)
    scratch_ramp.location = (-690, -520)
    dark_pos = float(np.random.uniform(0.44, 0.52))
    light_pos = float(np.random.uniform(0.56, 0.70))
    scratch_ramp.color_ramp.elements[0].position = dark_pos
    scratch_ramp.color_ramp.elements[1].position = light_pos
    detail["scratch_ramp_dark_position"] = dark_pos
    detail["scratch_ramp_light_position"] = light_pos

    scratch_bump = material.new_node("ShaderNodeBump", add_surface_wear_detail.__name__)
    scratch_bump.location = (-460, -520)
    set_node_input_value(scratch_bump, ["Strength"], detail["scratch_bump_strength"])
    set_node_input_value(scratch_bump, ["Distance"], detail["scratch_bump_distance"])

    try:
        material.link(get_socket(texcoord.outputs, ["Object", "Generated"]), get_socket(roughness_noise.inputs, ["Vector"]))
        material.link(get_socket(roughness_noise.outputs, ["Fac"]), get_socket(roughness_map.inputs, ["Value"]))

        roughness_input = get_socket(principled.inputs, ["Roughness"])
        roughness_output = get_socket(roughness_map.outputs, ["Result"])
        if roughness_input is not None and roughness_output is not None:
            remove_input_links(material, roughness_input)
            material.link(roughness_output, roughness_input)

        material.link(get_socket(texcoord.outputs, ["Object", "Generated"]), get_socket(scratch_wave.inputs, ["Vector"]))
        material.link(get_socket(scratch_wave.outputs, ["Fac"]), get_socket(scratch_ramp.inputs, ["Fac"]))
        material.link(get_socket(scratch_ramp.outputs, ["Color"]), get_socket(scratch_bump.inputs, ["Height"]))

        normal_input = get_socket(principled.inputs, ["Normal"])
        scratch_normal = get_socket(scratch_bump.outputs, ["Normal"])
        previous_normal = None
        if normal_input is not None and normal_input.links:
            previous_normal = normal_input.links[0].from_socket
        if normal_input is not None and scratch_normal is not None:
            remove_input_links(material, normal_input)
            if previous_normal is not None:
                previous_normal_input = get_socket(scratch_bump.inputs, ["Normal"])
                if previous_normal_input is not None:
                    material.link(previous_normal, previous_normal_input)
            material.link(scratch_normal, normal_input)
    except Exception as exc:
        material.blender_obj["surface_wear_warning"] = f"{type(exc).__name__}: {exc}"

    for key, value in detail.items():
        material.blender_obj[key] = value
    return detail


def make_random_material(
    name: str,
    texture_paths: Sequence[Path],
    metallic_range: Tuple[float, float],
    roughness_range: Tuple[float, float],
    brushed_metal: bool = False,
    surface_wear: bool = False,
) -> Any:
    material = bproc.material.create(name)

    if texture_paths:
        image = bpy.data.images.load(filepath=str(random.choice(texture_paths)), check_existing=True)
        set_shader_value(material, ["Base Color"], image)
    else:
        if brushed_metal:
            color = random_neutral_metal_color()
        else:
            color = np.append(np.random.uniform(0.08, 0.92, size=3), 1.0).tolist()
        set_shader_value(material, ["Base Color"], color)

    metallic = float(np.random.uniform(*metallic_range))
    roughness = float(np.random.uniform(*roughness_range))
    specular = float(np.random.uniform(0.25, 0.8))
    set_shader_value(material, ["Metallic"], metallic)
    set_shader_value(material, ["Roughness"], roughness)
    set_shader_value(material, ["Specular IOR Level", "Specular"], specular)
    material.blender_obj["profile"] = "random_pbr"
    material.blender_obj["brushed_metal"] = False
    material.blender_obj["metallic"] = metallic
    material.blender_obj["roughness"] = roughness
    material.blender_obj["specular"] = specular

    if brushed_metal:
        add_brushed_metal_detail(material)
    if surface_wear:
        add_surface_wear_detail(material, roughness)
    return material


def assign_random_material(
    obj: bproc.types.MeshObject,
    texture_paths: Sequence[Path],
    name: str,
    metallic_range: Tuple[float, float],
    roughness_range: Tuple[float, float],
    brushed_metal: bool = False,
    surface_wear: bool = False,
) -> Any:
    if texture_paths:
        try:
            if not obj.has_uv_mapping():
                obj.add_uv_mapping("smart")
        except Exception:
            pass
    material = make_random_material(
        name,
        texture_paths,
        metallic_range,
        roughness_range,
        brushed_metal=brushed_metal,
        surface_wear=surface_wear,
    )
    obj.replace_materials(material)
    return material


def load_scene_styles(scene_style_dir: Path) -> List[Dict[str, Any]]:
    if not scene_style_dir.exists():
        return []

    styles = []
    direct_textures = [p for p in scene_style_dir.iterdir() if p.is_file() and p.suffix.lower() in TEXTURE_SUFFIXES]
    if direct_textures:
        styles.append({"name": scene_style_dir.name, "textures": sorted(direct_textures)})

    for subdir in sorted(p for p in scene_style_dir.iterdir() if p.is_dir()):
        textures = list_files(subdir, TEXTURE_SUFFIXES)
        if textures:
            styles.append({"name": subdir.name, "textures": textures})
    return styles


def create_open_box(args: argparse.Namespace) -> List[bproc.types.MeshObject]:
    length = args.bin_length
    width = args.bin_width
    height = args.bin_height
    thick = args.wall_thickness
    bottom_z = thick / 2.0
    wall_z = thick + height / 2.0

    parts = []

    bottom = bproc.object.create_primitive(
        "CUBE",
        scale=[length / 2.0 + thick, width / 2.0 + thick, thick / 2.0],
        location=[0.0, 0.0, bottom_z],
    )
    bottom.set_name("receiver_box_bottom")
    parts.append(bottom)

    wall_specs = [
        ("receiver_box_y_pos", [0.0, width / 2.0 + thick / 2.0, wall_z], [length / 2.0 + thick, thick / 2.0, height / 2.0]),
        ("receiver_box_y_neg", [0.0, -width / 2.0 - thick / 2.0, wall_z], [length / 2.0 + thick, thick / 2.0, height / 2.0]),
        ("receiver_box_x_pos", [length / 2.0 + thick / 2.0, 0.0, wall_z], [thick / 2.0, width / 2.0, height / 2.0]),
        ("receiver_box_x_neg", [-length / 2.0 - thick / 2.0, 0.0, wall_z], [thick / 2.0, width / 2.0, height / 2.0]),
    ]
    for name, location, scale in wall_specs:
        wall = bproc.object.create_primitive("CUBE", scale=scale, location=location)
        wall.set_name(name)
        parts.append(wall)

    for obj in parts:
        set_cp(obj, "category_id", BACKGROUND_CATEGORY_ID)
        obj.blender_obj.hide_render = True #如果取消注释，则会让盒子透明化
        obj.enable_rigidbody(
            active=False,
            collision_shape="BOX",
            collision_margin=0.0005,
            friction=0.95,
        )
    return parts


def create_static_scene(args: argparse.Namespace) -> Dict[str, Any]:
    wall_distance = max(1.25, args.camera_radius_max + 0.30,
                        args.floor_size_x / 2.0 + 0.50, args.floor_size_y / 2.0 + 0.50)
    wall_height = 1.25
    wall_thick = 0.01

    if args.floor_size_x <= 0.0 or args.floor_size_y <= 0.0 or args.floor_thickness <= 0.0:
        raise ValueError("--floor_size_x, --floor_size_y and --floor_thickness must be positive")

    # A finite slab: its upper face is z=0, exactly where the old plane was.
    # The box therefore keeps resting on the tabletop, while its edge can
    # reveal the HDRI/world background behind it.
    floor = bproc.object.create_primitive(
        "CUBE",
        scale=[args.floor_size_x / 2.0, args.floor_size_y / 2.0, args.floor_thickness / 2.0],
        location=[0.0, 0.0, -args.floor_thickness / 2.0],
    )
    floor.set_name("scene_tabletop")
    set_cp(floor, "category_id", BACKGROUND_CATEGORY_ID)
    floor.enable_rigidbody(active=False, collision_shape="BOX", collision_margin=0.0005, friction=0.9)

    background_floor = None
    background_z = -args.floor_thickness - args.background_floor_gap
    if args.background_mode == "backgrounds":
        if not math.isfinite(args.background_floor_gap) or args.background_floor_gap <= 0.0:
            raise ValueError("--background_floor_gap must be positive and finite")
        # 下层仅作视觉背景；上层桌面保持 z=0，不改变箱子和零件的物理坐标。
        background_floor = bproc.object.create_primitive(
            "PLANE", scale=[wall_distance, wall_distance, 1.0],
            location=[0.0, 0.0, background_z],
        )
        background_floor.set_name("scene_background_floor")
        set_cp(background_floor, "category_id", BACKGROUND_CATEGORY_ID)

    room_walls = []
    # HDRI 模式不创建图片背景平面和房间墙壁。
    if args.background_mode == "backgrounds" and not args.disable_walls:
        wall_specs = [
            ("scene_wall_y_pos", [0.0, wall_distance, wall_height / 2.0], [wall_distance, wall_thick / 2.0, wall_height / 2.0]),
            ("scene_wall_y_neg", [0.0, -wall_distance, wall_height / 2.0], [wall_distance, wall_thick / 2.0, wall_height / 2.0]),
            ("scene_wall_x_pos", [wall_distance, 0.0, wall_height / 2.0], [wall_thick / 2.0, wall_distance, wall_height / 2.0]),
            ("scene_wall_x_neg", [-wall_distance, 0.0, wall_height / 2.0], [wall_thick / 2.0, wall_distance, wall_height / 2.0]),
        ]
        for name, location, scale in wall_specs:
            location[2] += background_z
            wall = bproc.object.create_primitive("CUBE", scale=scale, location=location)
            wall.set_name(name)
            set_cp(wall, "category_id", BACKGROUND_CATEGORY_ID)
            room_walls.append(wall)

    box_parts = create_open_box(args)
    return {"floor": floor, "background_floor": background_floor, "room_walls": room_walls, "box_parts": box_parts}


def set_scene_hdri(hdri_path: Path, strength: float) -> None:
    """复用世界节点，避免每一帧新增环境贴图、映射和坐标节点。"""
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = next(node for node in nodes if node.type == "BACKGROUND")
    environment = nodes.get("synmech_hdri_environment")
    if environment is None:
        environment = nodes.new("ShaderNodeTexEnvironment")
        environment.name = "synmech_hdri_environment"
        mapping = nodes.new("ShaderNodeMapping")
        mapping.name = "synmech_hdri_mapping"
        coordinates = nodes.new("ShaderNodeTexCoord")
        coordinates.name = "synmech_hdri_coordinates"
        links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    links.new(environment.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = strength


def randomize_static_scene(
    static_scene: Dict[str, Any],
    scene_styles: Sequence[Dict[str, Any]],
    ground_textures: Sequence[Path],
    container_textures: Sequence[Path],
    hdri_files: Sequence[Path],
    image_id: int,
    args: argparse.Namespace,
) -> str:
    style = random.choice(scene_styles) if args.background_mode == "backgrounds" else {"name": "hdris", "textures": []}
    scene_textures = style["textures"]
    box_textures = container_textures or ground_textures

    assign_random_material(
        static_scene["floor"],
        ground_textures,
        f"floor_mat_{image_id:06d}",
        metallic_range=(0.0, 0.12),
        roughness_range=(0.35, 1.0),
    )
    if static_scene["background_floor"] is not None:
        assign_random_material(
            static_scene["background_floor"], scene_textures,
            f"background_floor_mat_{image_id:06d}",
            metallic_range=(0.0, 0.08), roughness_range=(0.45, 1.0),
        )
    for index, wall in enumerate(static_scene["room_walls"]):
        assign_random_material(
            wall,
            scene_textures,
            f"room_wall_mat_{image_id:06d}_{index}",
            metallic_range=(0.0, 0.08),
            roughness_range=(0.45, 1.0),
        )
    # Container: use one shared material for all 5 box parts
    box_material = make_random_material(
        f"box_mat_{image_id:06d}",
        box_textures,
        metallic_range=(0.0, 0.25),
        roughness_range=(0.25, 0.95),
    )

    for box_part in static_scene["box_parts"]:
        if box_textures:
            try:
                if not box_part.has_uv_mapping():
                    box_part.add_uv_mapping("smart")
            except Exception:
                pass
        box_part.replace_materials(box_material)

    if args.background_mode == "hdris":
        hdri_path = random.choice(hdri_files)
        strength = float(np.random.uniform(args.hdri_strength_min, args.hdri_strength_max))
        set_scene_hdri(hdri_path, strength)
    else:
        # 图片背景模式使用常量环境光和独立灯光。
        bproc.renderer.set_world_background(
            np.random.uniform([0.02, 0.02, 0.02], [0.10, 0.10, 0.10]).tolist(),
            strength=float(np.random.uniform(0.02, 0.17)),
        )
    return str(style["name"])


def configure_physics(args: argparse.Namespace) -> None:
    bpy.context.scene.gravity = (0.0, 0.0, -9.81 * args.gravity_scale)
    if bpy.context.scene.rigidbody_world is not None:
        bpy.context.scene.rigidbody_world.enabled = True


def configure_camera_and_renderer(args: argparse.Namespace) -> None:
    cx = args.width / 2.0 if args.cx < 0 else args.cx
    cy = args.height / 2.0 if args.cy < 0 else args.cy
    k_matrix = np.array([[args.fx, 0.0, cx], [0.0, args.fy, cy], [0.0, 0.0, 1.0]])

    bproc.camera.set_intrinsics_from_K_matrix(k_matrix, args.width, args.height)
    bproc.camera.set_resolution(args.width, args.height)
    bpy.context.scene.camera.data.clip_start = 0.01
    bpy.context.scene.camera.data.clip_end = 10.0

    bproc.renderer.set_max_amount_of_samples(args.samples)
    try:
        bproc.renderer.set_denoiser("OPTIX")
    except Exception:
        pass
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "category_id", "name"],
        default_values={"category_id": BACKGROUND_CATEGORY_ID, "name": "background"},
    )


def load_joined_model(model_path: Path, category_id: int, model_scale: float) -> bproc.types.MeshObject:
    loaded_objects = bproc.loader.load_obj(str(model_path))
    mesh_objects = [obj for obj in loaded_objects if isinstance(obj, bproc.types.MeshObject)]
    if not mesh_objects:
        raise RuntimeError(f"No mesh object loaded from {model_path}")

    base_obj = mesh_objects[0]
    if len(mesh_objects) > 1:
        base_obj.join_with_other_objects(mesh_objects[1:])

    if model_scale != 1.0:
        base_obj.set_scale([model_scale, model_scale, model_scale])

    base_obj.set_name(f"template_{category_id:06d}_{model_path.stem}")
    set_cp(base_obj, "category_id", category_id)
    set_cp(base_obj, "model_name", model_path.stem)
    set_cp(base_obj, "semantic_name", semantic_name_from_stem(model_path.stem))
    set_cp(base_obj, "source_path", model_path.name)
    set_cp(base_obj, "is_part_template", True)
    base_obj.hide(True)
    return base_obj


def read_label_maps(dataset_dir: Path) -> Dict[str, Any]:
    labels_path = dataset_dir / "segmentation_labels.json"
    if not labels_path.exists():
        return {}
    with labels_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_part_templates(parts_dir: Path, model_scale: float, dataset_dir: Path = None) -> List[Dict[str, Any]]:
    """Reuse saved model IDs and reserve all historical IDs, including absent models.

    A model is identified by its source path. Renaming or moving it creates a new
    category; replacing its contents at the same path keeps the existing category.
    """
    model_paths = sorted(path for path in parts_dir.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES)
    if not model_paths:
        raise FileNotFoundError(f"No CAD files found in: {parts_dir}")

    existing = read_label_maps(dataset_dir) if dataset_dir is not None else {}
    categories_by_path = {}
    for category_key, label in existing.items():
        if int(category_key) == BACKGROUND_CATEGORY_ID:
            continue
        source_path = Path(label["source_path"])
        if source_path in categories_by_path:
            raise ValueError(f"Multiple saved category IDs for model: {source_path}")
        categories_by_path[source_path] = int(category_key)
    next_category_id = max((int(key) for key in existing), default=BACKGROUND_CATEGORY_ID) + 1

    templates = []
    for model_path in model_paths:
        relative_path = model_path.relative_to(parts_dir)
        category_id = categories_by_path.get(relative_path)
        if category_id is None:
            category_id = next_category_id
            next_category_id += 1
        if not 1 <= category_id <= 65535:
            raise ValueError("Category IDs must fit in uint16 segmentation PNGs (1..65535).")
        saved_label = existing.get(str(category_id))
        template = load_joined_model(model_path, category_id, model_scale)
        template_info = {
            "category_id": category_id,
            "model_name": saved_label["name"] if saved_label else model_path.stem,
            "semantic_name": saved_label["semantic_name"] if saved_label else semantic_name_from_stem(model_path.stem),
            "source_path": saved_label["source_path"] if saved_label else relative_path.as_posix(),
            "object": template,
        }
        # Keep Blender properties and metadata consistent with the saved mapping.
        for name in ("model_name", "semantic_name", "source_path"):
            set_cp(template, name, template_info[name])
        templates.append(template_info)
    return templates


def write_label_maps(dataset_dir: Path, templates: Sequence[Dict[str, Any]]) -> None:
    existing = read_label_maps(dataset_dir)
    labels = dict(existing)
    current_labels = {
        "0": {
            "name": "background_scene_and_receiver_box",
            "semantic_name": "background",
            "is_target": False,
        }
    }
    for item in templates:
        category_key = str(item["category_id"])
        label = {
            "name": item["model_name"],
            "semantic_name": item["semantic_name"],
            "source_path": item["source_path"],
            "is_target": True,
        }
        if category_key in current_labels and current_labels[category_key] != label:
            raise ValueError(f"Multiple models assigned to category {category_key}.")
        current_labels[category_key] = label
    for category_key, label in current_labels.items():
        if category_key in existing and existing[category_key] != label:
            raise ValueError(f"Cannot relabel existing category {category_key}: "
                             f"{existing[category_key]['name']} -> {label['name']}. "
                             "Use a new --dataset-name to avoid relabeling old images.")
        labels[category_key] = label
    labels_path = dataset_dir / "segmentation_labels.json"
    if labels != existing:
        # Replace only after a complete write, so an interrupted write keeps the old map.
        temporary_path = labels_path.with_suffix(".json.tmp")
        save_json(temporary_path, labels)
        temporary_path.replace(labels_path)


def set_part_custom_properties(obj: bproc.types.MeshObject, template_info: Dict[str, Any], instance_index: int) -> None:
    set_cp(obj, "category_id", int(template_info["category_id"]))
    set_cp(obj, "model_name", template_info["model_name"])
    set_cp(obj, "semantic_name", template_info["semantic_name"])
    set_cp(obj, "source_path", template_info["source_path"])
    set_cp(obj, "instance_index", int(instance_index))
    set_cp(obj, "is_generated_part", True)


def get_drop_xy_range(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    min_x = -args.bin_length / 2.0 + args.drop_xy_margin
    max_x = args.bin_length / 2.0 - args.drop_xy_margin
    min_y = -args.bin_width / 2.0 + args.drop_xy_margin
    max_y = args.bin_width / 2.0 - args.drop_xy_margin

    if min_x >= max_x:
        min_x, max_x = -args.bin_length * 0.35, args.bin_length * 0.35
    if min_y >= max_y:
        min_y, max_y = -args.bin_width * 0.35, args.bin_width * 0.35

    return min_x, max_x, min_y, max_y


def sample_random_drop_location(index: int, args: argparse.Namespace) -> List[float]:
    min_x, max_x, min_y, max_y = get_drop_xy_range(args)

    return [
        float(np.random.uniform(min_x, max_x)),
        float(np.random.uniform(min_y, max_y)),
        float(
            np.random.uniform(args.drop_height_min, args.drop_height_max)
            + min(index, 10) * args.initial_z_spacing
        ),
    ]


def get_world_aabb(obj: bproc.types.MeshObject) -> Tuple[np.ndarray, np.ndarray]:
    bpy.context.view_layer.update()

    blender_obj = obj.blender_obj
    world_corners = [
        blender_obj.matrix_world @ Vector(corner)
        for corner in blender_obj.bound_box
    ]

    corners_np = np.array([[corner.x, corner.y, corner.z] for corner in world_corners], dtype=float)
    return np.min(corners_np, axis=0), np.max(corners_np, axis=0)


def aabb_overlaps(
    a_min: np.ndarray,
    a_max: np.ndarray,
    b_min: np.ndarray,
    b_max: np.ndarray,
    margin: float = 0.0,
) -> bool:
    return bool(
        (a_min[0] <= b_max[0] + margin and a_max[0] + margin >= b_min[0]) and
        (a_min[1] <= b_max[1] + margin and a_max[1] + margin >= b_min[1]) and
        (a_min[2] <= b_max[2] + margin and a_max[2] + margin >= b_min[2])
    )


def aabb_overlaps_any(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    placed_aabbs: Sequence[Tuple[np.ndarray, np.ndarray]],
    margin: float,
) -> bool:
    for other_min, other_max in placed_aabbs:
        if aabb_overlaps(aabb_min, aabb_max, other_min, other_max, margin=margin):
            return True
    return False


def sample_initial_drop_pose(
    obj: bproc.types.MeshObject,
    index: int,
    placed_aabbs: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> Tuple[bool, Tuple[np.ndarray, np.ndarray]]:
    for try_id in range(args.max_initial_pose_tries):
        obj.set_location(sample_random_drop_location(index, args))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

        aabb_min, aabb_max = get_world_aabb(obj)
        if not aabb_overlaps_any(
            aabb_min,
            aabb_max,
            placed_aabbs,
            margin=args.init_overlap_margin,
        ):
            if try_id > 0:
                print(f"[InitPose] {obj.get_name()} placed after {try_id + 1} tries.")
            return True, (aabb_min, aabb_max)

    print(
        f"[WARN] Could not find non-overlapping initial pose for {obj.get_name()} "
        f"after {args.max_initial_pose_tries} tries."
    )
    return False, (np.zeros(3, dtype=float), np.zeros(3, dtype=float))


def configure_active_rigidbody(obj: bproc.types.MeshObject, args: argparse.Namespace) -> None:
    obj.enable_rigidbody(
        active=True,
        collision_shape=args.collision_shape,
        collision_margin=0.0005,
        mass_factor=1200.0,
        friction=float(np.random.uniform(0.55, 0.95)),
        linear_damping=args.linear_damping,
        angular_damping=args.angular_damping,
    )


def instantiate_parts(
    templates: Sequence[Dict[str, Any]],
    part_textures: Sequence[Path],
    args: argparse.Namespace,
    image_id: int,
) -> List[bproc.types.MeshObject]:
    num_parts = random.randint(args.min_parts, args.max_parts)
    selected = random.choices(list(templates), k=num_parts)

    active_parts = []
    created_parts = []
    placed_aabbs = []

    try:
        for index, template_info in enumerate(selected):
            obj = template_info["object"].duplicate()
            created_parts.append(obj)

            obj.hide(False)
            obj.set_name(f"{template_info['model_name']}_im{image_id:06d}_inst{index:02d}")
            set_part_custom_properties(obj, template_info, index)
            assign_random_material(
                obj,
                part_textures,
                f"part_mat_{image_id:06d}_{index:02d}",
                metallic_range=(0.80, 1.0),
                roughness_range=(0.16, 0.36),
                brushed_metal=True,
                surface_wear=not args.disable_surface_wear,
            )

            success, accepted_aabb = sample_initial_drop_pose(
                obj=obj,
                index=index,
                placed_aabbs=placed_aabbs,
                args=args,
            )
            if not success:
                raise RuntimeError(
                    f"Failed to place part {index} without initial overlap. "
                    f"Try reducing --min_parts/--max_parts, increasing --drop_height_max, "
                    f"reducing --init_overlap_margin, or increasing --max_initial_pose_tries."
                )

            configure_active_rigidbody(obj, args)
            active_parts.append(obj)
            placed_aabbs.append(accepted_aabb)

        return active_parts

    except Exception:
        delete_entities(created_parts)
        raise

def simulate_drop(parts: Sequence[bproc.types.MeshObject], args: argparse.Namespace) -> bool:
    try:
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=args.physics_min_time,
            max_simulation_time=args.physics_max_time,
            check_object_interval=1.0,
            substeps_per_frame=args.physics_substeps,
            solver_iters=args.physics_solver_iters,
        )
    except Exception as exc:
        print(f"[WARN] Physics simulation failed: {type(exc).__name__}: {exc}")
        return False

    xy_limit_x = args.bin_length * 0.5
    xy_limit_y = args.bin_width * 0.5
    for obj in parts:
        loc = np.array(obj.blender_obj.location, dtype=float)
        if abs(loc[0]) > xy_limit_x or abs(loc[1]) > xy_limit_y or loc[2] < -0.08 or loc[2] > 1.8:
            print(
                f"[Reject] {obj.get_name()} out of valid range: "
                f"loc=({loc[0]:.4f}, {loc[1]:.4f}, {loc[2]:.4f}), "
                f"xy_limit=({xy_limit_x:.4f}, {xy_limit_y:.4f})"
            )
            return False
    return True


def kelvin_to_rgb(temp_k: float) -> List[float]:
    temp = max(1000.0, min(40000.0, temp_k)) / 100.0

    if temp <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(temp) - 161.1195681661
        if temp <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        b = 255.0

    return [
        float(np.clip(r, 0.0, 255.0) / 255.0),
        float(np.clip(g, 0.0, 255.0) / 255.0),
        float(np.clip(b, 0.0, 255.0) / 255.0),
    ]


def random_light_color() -> List[float]:
    temp_k = float(np.random.uniform(2800.0, 8500.0))
    color = np.array(kelvin_to_rgb(temp_k), dtype=float)

    color *= np.random.uniform(0.90, 1.08, size=3)
    color = np.clip(color, 0.0, 1.0)

    return color.tolist()


def create_random_lights(image_id: int) -> List[Any]:
    lights = []

    selected_light_type = random.choices(
        ["POINT", "AREA", "SUN"],
        weights=[0.40, 0.45, 0.15],
        k=1,
    )[0]

    lighting_mode = random.choices(
        ["dark", "normal", "bright"],
        weights=[0.15, 0.70, 0.15],
        k=1,
    )[0]

    if lighting_mode == "dark":
        global_scale = float(np.random.uniform(0.35, 0.70))
    elif lighting_mode == "normal":
        global_scale = float(np.random.uniform(0.65, 1.00))
    else:
        global_scale = float(np.random.uniform(1.00, 1.20))

    if selected_light_type == "POINT":
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_name(f"point_light_{image_id:06d}")
        light.blender_obj["is_generated_light"] = True

        light.set_location(
            [
                float(np.random.uniform(-0.8, 0.8)),
                float(np.random.uniform(-0.8, 0.8)),
                float(np.random.uniform(0.55, 1.55)),
            ]
        )

        light.set_energy(float(np.random.uniform(80.0, 320.0) * global_scale))
        light.set_color(random_light_color())

        try:
            light.blender_obj.data.shadow_soft_size = float(np.random.uniform(0.03, 0.35))
        except Exception:
            pass

        lights.append(light)

    elif selected_light_type == "AREA":
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_name(f"area_light_{image_id:06d}")
        light.blender_obj["is_generated_light"] = True

        light.set_location(
            [
                float(np.random.uniform(-0.55, 0.55)),
                float(np.random.uniform(-0.55, 0.55)),
                float(np.random.uniform(0.70, 1.65)),
            ]
        )

        light.set_energy(float(np.random.uniform(50.0, 280.0) * global_scale))
        light.set_color(random_light_color())

        try:
            light.blender_obj.data.shape = random.choice(["SQUARE", "RECTANGLE", "DISK"])
            light.blender_obj.data.size = float(np.random.uniform(0.15, 0.80))
            if light.blender_obj.data.shape == "RECTANGLE":
                light.blender_obj.data.size_y = float(np.random.uniform(0.15, 0.80))
        except Exception:
            try:
                light.blender_obj.data.size = float(np.random.uniform(0.15, 0.80))
            except Exception:
                pass

        lights.append(light)

    elif selected_light_type == "SUN":
        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_name(f"sun_light_{image_id:06d}")
        light.blender_obj["is_generated_light"] = True

        light.set_rotation_euler(
            [
                float(np.random.uniform(-0.5, 0.5)),
                float(np.random.uniform(-0.5, 0.5)),
                float(np.random.uniform(-math.pi, math.pi)),
            ]
        )

        light.set_energy(float(np.random.uniform(0.2, 1.70) * global_scale))
        light.set_color(random_light_color())

        try:
            light.blender_obj.data.angle = float(np.random.uniform(0.01, 0.18))
        except Exception:
            pass

        lights.append(light)

    print(
        f"[Light] image={image_id:06d}, type={selected_light_type}, "
        f"mode={lighting_mode}, global_scale={global_scale:.3f}"
    )

    return lights


def sample_camera_pose(args: argparse.Namespace) -> np.ndarray:
    target = np.array([0.0, 0.0, args.wall_thickness + args.bin_height * 0.55], dtype=float)
    radius = float(np.random.uniform(args.camera_radius_min, args.camera_radius_max))
    azimuth = float(np.random.uniform(0.0, 2.0 * math.pi))
    elevation = math.radians(float(np.random.uniform(args.camera_elev_min, args.camera_elev_max)))

    location = np.array(
        [
            target[0] + radius * math.cos(elevation) * math.cos(azimuth),
            target[1] + radius * math.cos(elevation) * math.sin(azimuth),
            target[2] + radius * math.sin(elevation),
        ],
        dtype=float,
    )
    rotation = bproc.camera.rotation_from_forward_vec(
        target - location,
        inplane_rot=math.radians(float(np.random.uniform(-args.camera_inplane_deg, args.camera_inplane_deg))),
    )
    return bproc.math.build_transformation_mat(location, rotation)


def reset_keyframes_for_single_frame() -> None:
    try:
        bproc.utility.reset_keyframes()
    except Exception:
        for obj in bpy.context.scene.objects:
            obj.animation_data_clear()
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0
    bpy.context.scene.frame_set(0)


def clear_registered_output(key: str) -> None:
    if not GlobalStorage.is_in_storage("output"):
        return
    outputs = [output for output in GlobalStorage.get("output") if output.get("key") != key]
    GlobalStorage.set("output", outputs)


def visible_instance_stats(render_data: Dict[str, Any], active_parts: Sequence[bproc.types.MeshObject]) -> Tuple[int, Dict[str, int]]:
    instance_segmap = first_frame(render_data.get("instance_segmaps"))
    if instance_segmap is None:
        return 0, {}

    stats = {}
    visible_count = 0
    for obj in active_parts:
        instance_id = int(obj.blender_obj.pass_index)
        pixel_count = int(np.count_nonzero(instance_segmap == instance_id))
        stats[obj.get_name()] = pixel_count
        if pixel_count > 0:
            set_cp(obj, "visible_pixels", pixel_count)
        if pixel_count >= 1:
            visible_count += 1
    return visible_count, stats


def depth_valid_limit_m() -> float:
    camera = bpy.context.scene.camera
    if camera is not None and getattr(camera, "data", None) is not None:
        clip_end = float(camera.data.clip_end)
        if clip_end > 0.0:
            return clip_end
    return 1.0e6


def valid_depth_mask(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    return np.isfinite(depth) & (depth > 0.0) & (depth < depth_valid_limit_m())


def sanitize_depth_frame(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    sanitized = np.array(depth, dtype=np.float32, copy=True)
    sanitized[~valid_depth_mask(sanitized)] = 0.0
    return sanitized


def sanitize_render_depths(render_data: Dict[str, Any]) -> None:
    depths = render_data.get("depth")
    if depths is None:
        return
    if isinstance(depths, list):
        render_data["depth"] = [sanitize_depth_frame(depth) for depth in depths]
    else:
        render_data["depth"] = sanitize_depth_frame(depths)


def valid_depth_pixels(render_data: Dict[str, Any]) -> int:
    depth = first_frame(render_data.get("depth"))
    if depth is None:
        return 0
    return int(np.count_nonzero(valid_depth_mask(depth)))


def resolve_start_image_id(dataset_dir: Path, requested_id: int = None) -> int:
    """新文件夹可指定起点；续写时避开历史图片和记录中的编号。"""
    max_id = -1
    frame_count = 0
    chunks_dir = dataset_dir / "train_pbr"
    if chunks_dir.exists():
        for chunk in chunks_dir.iterdir():
            if not chunk.is_dir() or not chunk.name.isdigit():
                continue
            with (chunk / "scene_gt.json").open(encoding="utf-8") as handle:
                gt = json.load(handle)
            frame_count += len(gt)
            max_id = max(max_id, max((int(key) for key in gt), default=-1))
            # 未完成写入的图片也不能被覆盖。
            for subdir in ("rgb", "depth", "seg_category", "seg_instance", "seg_instance_labels"):
                folder = chunk / subdir
                if folder.exists():
                    max_id = max(max_id, max((int(p.stem) for p in folder.iterdir()
                                             if p.is_file() and p.stem.isdigit()), default=-1))
    records_path = dataset_dir / "meta" / "pose_records.jsonl"
    if records_path.exists():
        with records_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    max_id = max(max_id, int(json.loads(line)["image_id"]))
    # 兼容旧版每个 chunk 都从 0 编号的数据；新记录从已有帧总数之后继续。
    next_id = max(max_id + 1, frame_count)
    if requested_id is not None:
        if requested_id < next_id:
            raise ValueError(f"--start-image-id must be >= {next_id} for this dataset; "
                             "choose a new --dataset-name to start a separate sequence")
        return requested_id
    return next_id


def get_next_bop_indices(dataset_dir: Path, frames_per_chunk: int) -> Tuple[int, int]:
    """Return the next BOP chunk/frame indices for append mode."""
    train_pbr = dataset_dir / "train_pbr"
    if not train_pbr.exists():
        return 0, 0

    chunk_ids = sorted(int(p.name) for p in train_pbr.iterdir() if p.is_dir() and p.name.isdigit())
    if not chunk_ids:
        return 0, 0

    chunk_id = chunk_ids[-1]
    scene_gt = train_pbr / f"{chunk_id:06d}" / "scene_gt.json"
    frame_id = 0
    if scene_gt.exists():
        with scene_gt.open(encoding="utf-8") as handle:
            records = json.load(handle)
        frame_id = max((int(k) for k in records), default=-1) + 1

    if frame_id >= frames_per_chunk:
        return chunk_id + 1, 0
    return chunk_id, frame_id


def write_uint16_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    clipped = np.clip(image, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(clipped).save(path)


def write_label_vis(path: Path, label_map: np.ndarray) -> None:
    ensure_dir(path.parent)
    labels = np.asarray(label_map, dtype=np.int64)
    vis = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for label_id in np.unique(labels):
        label_id = int(label_id)
        if label_id == 0:
            continue
        vis[labels == label_id] = [
            50 + (label_id * 53) % 206,
            50 + (label_id * 97) % 206,
            50 + (label_id * 193) % 206,
        ]
    Image.fromarray(vis).save(path)


def write_depth_vis(path: Path, depth_m: np.ndarray) -> None:
    ensure_dir(path.parent)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = valid_depth_mask(depth)

    vis_gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        near, far = np.percentile(depth[valid], [1.0, 99.0])
        if far <= near:
            far = near + 1e-6
        vis_gray[valid] = np.clip((depth[valid] - near) / (far - near) * 255.0, 0, 255).astype(np.uint8)

    Image.fromarray(vis_gray).save(path)


def filter_part_instance_segmap(
    instance_segmap: np.ndarray,
    active_parts: Sequence[bproc.types.MeshObject],
) -> np.ndarray:
    labels = np.asarray(instance_segmap)
    part_instance_ids = [int(obj.blender_obj.pass_index) for obj in active_parts]
    if not part_instance_ids:
        return np.zeros_like(labels)
    return np.where(np.isin(labels, part_instance_ids), labels, 0).astype(labels.dtype, copy=False)


def write_segmentation_outputs(
    dataset_dir: Path,
    chunk_id: int,
    frame_id: int,
    render_data: Dict[str, Any],
    active_parts: Sequence[bproc.types.MeshObject],
) -> None:
    chunk_dir = dataset_dir / "train_pbr" / f"{chunk_id:06d}"
    category_segmap = first_frame(render_data.get("category_id_segmaps"))
    instance_segmap = first_frame(render_data.get("instance_segmaps"))

    if category_segmap is not None:
        write_uint16_png(chunk_dir / "seg_category" / f"{frame_id:06d}.png", category_segmap)
        write_label_vis(chunk_dir / "seg_category_vis" / f"{frame_id:06d}.png", category_segmap)
    if instance_segmap is not None:
        instance_segmap = filter_part_instance_segmap(instance_segmap, active_parts)
        write_uint16_png(chunk_dir / "seg_instance" / f"{frame_id:06d}.png", instance_segmap)
        write_label_vis(chunk_dir / "seg_instance_vis" / f"{frame_id:06d}.png", instance_segmap)

    depth = first_frame(render_data.get("depth"))
    if depth is not None:
        write_depth_vis(chunk_dir / "depth_vis" / f"{frame_id:06d}.png", depth)

    instance_labels = []
    for obj in active_parts:
        instance_id = int(obj.blender_obj.pass_index)
        pixel_count = int(np.count_nonzero(instance_segmap == instance_id)) if instance_segmap is not None else 0
        
        # 加这一行拦截：如果像素为 0，就不写入 JSON
        if pixel_count <= 0:
            continue

        instance_labels.append(
            {
                "instance_id": instance_id,
                "category_id": int(obj.get_cp("category_id")),
                "object_name": obj.get_name(),
                "model_name": obj.get_cp("model_name"),
                "semantic_name": obj.get_cp("semantic_name"),
                "visible_pixels_in_instance_segmap": pixel_count,
            }
        )
    save_json(chunk_dir / "seg_instance_labels" / f"{frame_id:06d}.json", instance_labels)


def get_world_to_camera_opencv() -> Matrix:
    cam2world = Matrix(
        _WriterUtility.get_cam_attribute(
            bpy.context.scene.camera,
            "cam2world_matrix",
            local_frame_change=["X", "-Y", "-Z"],
        )
    )
    return cam2world.inverted()


def build_pose_record(
    image_id: int,
    chunk_id: int,
    frame_id: int,
    active_parts: Sequence[bproc.types.MeshObject],
    scene_style_name: str,
    visible_pixels: Dict[str, int],
) -> Dict[str, Any]:
    world_to_cam = get_world_to_camera_opencv()
    objects = []
    for obj in active_parts:
        m2w = Matrix(obj.get_local2world_mat())
        m2c = world_to_cam @ m2w
        objects.append(
            {
                "object_name": obj.get_name(),
                "category_id": int(obj.get_cp("category_id")),
                "model_name": obj.get_cp("model_name"),
                "semantic_name": obj.get_cp("semantic_name"),
                "cam_R_m2c": [float(v) for v in np.array(m2c.to_quaternion().to_matrix()).reshape(-1)],
                "cam_t_m2c_m": [float(v) for v in np.array(m2c.to_translation()).reshape(-1)],
                "cam_t_m2c_mm": [float(v) * 1000.0 for v in np.array(m2c.to_translation()).reshape(-1)],
                "world_location_m": [float(v) for v in obj.blender_obj.location],
                "visible_pixels_in_instance_segmap": int(visible_pixels.get(obj.get_name(), 0)),
            }
        )

    return {
        "image_id": image_id,
        "bop_chunk_id": chunk_id,
        "bop_frame_id": frame_id,
        "scene_style": scene_style_name,
        "objects": objects,
    }


def summarize_numeric(values: Sequence[Any]) -> Dict[str, Any]:
    cleaned = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            cleaned.append(number)

    if not cleaned:
        return {"count": 0}

    arr = np.asarray(cleaned, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def summarize_categorical(values: Sequence[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def get_primary_material(obj: bproc.types.MeshObject) -> Any:
    try:
        if obj.blender_obj.data.materials:
            return obj.blender_obj.data.materials[0]
    except Exception:
        pass
    return None


def read_material_quality(obj: bproc.types.MeshObject) -> Dict[str, Any]:
    mat = get_primary_material(obj)
    record = {"material_name": None, "profile": "unknown", "brushed_metal": False}
    if mat is None:
        return record

    record["material_name"] = mat.name
    for key in (
        "profile",
        "brushed_metal",
        "metallic",
        "roughness",
        "specular",
        "anisotropic",
        "anisotropic_rotation",
        "line_scale",
        "line_distortion",
        "line_direction",
        "bump_strength",
        "bump_distance",
        "ramp_dark_position",
        "ramp_light_position",
        "surface_profile",
        "base_roughness_before_surface_wear",
        "roughness_noise_scale",
        "roughness_noise_detail",
        "roughness_noise_roughness",
        "roughness_noise_strength",
        "roughness_map_min",
        "roughness_map_max",
        "scratch_scale",
        "scratch_distortion",
        "scratch_direction",
        "scratch_bump_strength",
        "scratch_bump_distance",
        "scratch_ramp_dark_position",
        "scratch_ramp_light_position",
        "dirt_strength",
        "oil_strength",
        "brushed_detail_warning",
        "surface_wear_warning",
    ):
        try:
            if key in mat:
                value = mat[key]
                if isinstance(value, np.generic):
                    value = value.item()
                record[key] = value
        except Exception:
            continue
    return record


def depth_quality_stats(render_data: Dict[str, Any], depth_pixels: int) -> Dict[str, Any]:
    depth = first_frame(render_data.get("depth"))
    if depth is None:
        return {"valid_pixels": int(depth_pixels), "valid_fraction": 0.0}

    depth = np.asarray(depth, dtype=np.float32)
    valid = valid_depth_mask(depth)
    valid_values = depth[valid]
    stats = {
        "total_pixels": int(depth.size),
        "valid_pixels": int(depth_pixels),
        "valid_fraction": float(depth_pixels / depth.size) if depth.size else 0.0,
    }
    if valid_values.size > 0:
        stats.update(
            {
                "min_m": float(np.min(valid_values)),
                "max_m": float(np.max(valid_values)),
                "p01_m": float(np.percentile(valid_values, 1)),
                "p50_m": float(np.percentile(valid_values, 50)),
                "p99_m": float(np.percentile(valid_values, 99)),
            }
        )
    return stats


def segmentation_quality_stats(render_data: Dict[str, Any], active_parts: Sequence[bproc.types.MeshObject]) -> Dict[str, Any]:
    category_segmap = first_frame(render_data.get("category_id_segmaps"))
    instance_segmap = first_frame(render_data.get("instance_segmaps"))
    stats = {}

    if category_segmap is not None:
        category_segmap = np.asarray(category_segmap)
        foreground = category_segmap != BACKGROUND_CATEGORY_ID
        stats.update(
            {
                "category_total_pixels": int(category_segmap.size),
                "category_foreground_pixels": int(np.count_nonzero(foreground)),
                "category_foreground_fraction": float(np.count_nonzero(foreground) / category_segmap.size)
                if category_segmap.size
                else 0.0,
                "category_label_count": int(len(np.unique(category_segmap))),
            }
        )

    if instance_segmap is not None:
        raw_instance_segmap = np.asarray(instance_segmap)
        instance_segmap = filter_part_instance_segmap(raw_instance_segmap, active_parts)
        part_instance_ids = [int(obj.blender_obj.pass_index) for obj in active_parts]
        visible_part_instance_ids = [
            instance_id for instance_id in part_instance_ids if int(np.count_nonzero(instance_segmap == instance_id)) > 0
        ]
        stats.update(
            {
                "instance_total_pixels": int(instance_segmap.size),
                "instance_label_count": int(len(np.unique(instance_segmap))),
                "raw_instance_label_count": int(len(np.unique(raw_instance_segmap))),
                "part_instance_count": int(len(part_instance_ids)),
                "visible_part_instance_count": int(len(visible_part_instance_ids)),
            }
        )

    return stats


def build_quality_record(
    image_id: int,
    attempt: int,
    chunk_id: int,
    frame_id: int,
    active_parts: Sequence[bproc.types.MeshObject],
    scene_style_name: str,
    render_data: Dict[str, Any],
    visible_pixels: Dict[str, int],
    depth_pixels: int,
    min_visible_pixels: int,
) -> Dict[str, Any]:
    object_records = []
    visible_values = []
    for obj in active_parts:
        pixel_count = int(visible_pixels.get(obj.get_name(), 0))

        # 加这一行拦截：如果像素为0，就不写入json
        if pixel_count <= 0:
            continue

        visible_values.append(pixel_count)
        object_records.append(
            {
                "object_name": obj.get_name(),
                "category_id": int(obj.get_cp("category_id")),
                "model_name": obj.get_cp("model_name"),
                "visible_pixels": pixel_count,
                "strong_visible": bool(pixel_count >= min_visible_pixels),
                "material": read_material_quality(obj),
            }
        )

    strong_visible_count = sum(1 for value in visible_values if value >= min_visible_pixels)
    material_records = [record["material"] for record in object_records]

    return {
        "image_id": int(image_id),
        "attempt": int(attempt),
        "bop_chunk_id": int(chunk_id),
        "bop_frame_id": int(frame_id),
        "scene_style": scene_style_name,
        "part_count": int(len(active_parts)),
        "visible_part_count": int(sum(1 for value in visible_values if value > 0)),
        "strong_visible_part_count": int(strong_visible_count),
        "visible_pixels_summary": summarize_numeric(visible_values),
        "depth": depth_quality_stats(render_data, depth_pixels),
        "segmentation": segmentation_quality_stats(render_data, active_parts),
        "material_summary": {
            "brushed_metal_parts": int(
                sum(1 for record in material_records if bool(record.get("brushed_metal", False)))
            ),
            "surface_profile_counts": summarize_categorical(record.get("surface_profile") for record in material_records),
            "metallic": summarize_numeric(record.get("metallic") for record in material_records),
            "roughness": summarize_numeric(record.get("roughness") for record in material_records),
            "anisotropic": summarize_numeric(record.get("anisotropic") for record in material_records),
            "bump_strength": summarize_numeric(record.get("bump_strength") for record in material_records),
            "bump_distance": summarize_numeric(record.get("bump_distance") for record in material_records),
            "line_scale": summarize_numeric(record.get("line_scale") for record in material_records),
            "roughness_noise_strength": summarize_numeric(
                record.get("roughness_noise_strength") for record in material_records
            ),
            "roughness_map_min": summarize_numeric(record.get("roughness_map_min") for record in material_records),
            "roughness_map_max": summarize_numeric(record.get("roughness_map_max") for record in material_records),
            "scratch_bump_strength": summarize_numeric(
                record.get("scratch_bump_strength") for record in material_records
            ),
            "scratch_bump_distance": summarize_numeric(
                record.get("scratch_bump_distance") for record in material_records
            ),
            "dirt_strength": summarize_numeric(record.get("dirt_strength") for record in material_records),
            "oil_strength": summarize_numeric(record.get("oil_strength") for record in material_records),
        },
        "objects": object_records,
    }


def build_quality_report(stats: Dict[str, Any], quality_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    object_records = [obj for record in quality_records for obj in record.get("objects", [])]
    material_records = [obj.get("material", {}) for obj in object_records]
    attempts = int(stats.get("attempts", 0))
    accepted_images = int(stats.get("accepted_images", 0))

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generation_summary": dict(stats),
        "dataset_quality_summary": {
            "accepted_images": int(len(quality_records)),
            "acceptance_rate": float(accepted_images / attempts) if attempts else 0.0,
            "visible_parts_per_image": summarize_numeric(record.get("visible_part_count") for record in quality_records),
            "strong_visible_parts_per_image": summarize_numeric(
                record.get("strong_visible_part_count") for record in quality_records
            ),
            "depth_valid_fraction": summarize_numeric(
                record.get("depth", {}).get("valid_fraction") for record in quality_records
            ),
            "category_foreground_fraction": summarize_numeric(
                record.get("segmentation", {}).get("category_foreground_fraction") for record in quality_records
            ),
            "visible_pixels_per_object": summarize_numeric(obj.get("visible_pixels") for obj in object_records),
            "material": {
                "object_material_count": int(len(material_records)),
                "brushed_metal_object_count": int(
                    sum(1 for record in material_records if bool(record.get("brushed_metal", False)))
                ),
                "surface_profile_counts": summarize_categorical(
                    record.get("surface_profile") for record in material_records
                ),
                "metallic": summarize_numeric(record.get("metallic") for record in material_records),
                "roughness": summarize_numeric(record.get("roughness") for record in material_records),
                "anisotropic": summarize_numeric(record.get("anisotropic") for record in material_records),
                "bump_strength": summarize_numeric(record.get("bump_strength") for record in material_records),
                "bump_distance": summarize_numeric(record.get("bump_distance") for record in material_records),
                "line_scale": summarize_numeric(record.get("line_scale") for record in material_records),
                "roughness_noise_strength": summarize_numeric(
                    record.get("roughness_noise_strength") for record in material_records
                ),
                "roughness_map_min": summarize_numeric(record.get("roughness_map_min") for record in material_records),
                "roughness_map_max": summarize_numeric(record.get("roughness_map_max") for record in material_records),
                "scratch_bump_strength": summarize_numeric(
                    record.get("scratch_bump_strength") for record in material_records
                ),
                "scratch_bump_distance": summarize_numeric(
                    record.get("scratch_bump_distance") for record in material_records
                ),
                "dirt_strength": summarize_numeric(record.get("dirt_strength") for record in material_records),
                "oil_strength": summarize_numeric(record.get("oil_strength") for record in material_records),
            },
        },
        "image_quality_records": list(quality_records),
    }


def remove_blender_objects(blender_objects: Sequence[bpy.types.Object]) -> None:
    if not blender_objects:
        return

    seen = set()
    for obj in list(blender_objects):
        try:
            obj_name = obj.name
        except ReferenceError:
            continue
        if obj_name in seen or obj_name not in bpy.data.objects:
            continue
        seen.add(obj_name)
        try:
            object_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if object_data is not None and object_data.users == 0:
                if isinstance(object_data, bpy.types.Mesh):
                    bpy.data.meshes.remove(object_data)
                elif isinstance(object_data, bpy.types.Light):
                    bpy.data.lights.remove(object_data)
        except ReferenceError:
            pass
        except Exception as exc:
            print(f"[WARN] Could not delete object {obj_name}: {type(exc).__name__}: {exc}")


def delete_entities(entities: Sequence[Any]) -> None:
    remove_blender_objects([entity.blender_obj for entity in entities if hasattr(entity, "blender_obj")])


def cleanup_generated_attempt_objects() -> None:
    generated_objects = []
    for obj in list(bpy.data.objects):
        try:
            if obj.get("is_generated_part", False) or obj.get("is_generated_light", False):
                generated_objects.append(obj)
        except ReferenceError:
            continue
    remove_blender_objects(generated_objects)


def cleanup_unused_materials_and_images() -> None:
    for mat in list(bpy.data.materials):
        try:
            if mat.users == 0 and mat.name.startswith(GENERATED_MATERIAL_PREFIXES):
                bpy.data.materials.remove(mat)
        except ReferenceError:
            continue
        except Exception as exc:
            print(f"[WARN] Could not delete unused material: {type(exc).__name__}: {exc}")

    for image in list(bpy.data.images):
        try:
            image_path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
            if image.users == 0 and image_path and image_path.suffix.lower() in TEXTURE_SUFFIXES | HDRI_SUFFIXES:
                bpy.data.images.remove(image)
        except ReferenceError:
            continue
        except Exception as exc:
            print(f"[WARN] Could not delete unused image: {type(exc).__name__}: {exc}")


def prepare_output(args: argparse.Namespace) -> Tuple[Path, Path]:
    output_root = Path(args.output_dir).resolve()
    dataset_dir = (output_root / args.dataset_name).resolve()
    if dataset_dir == output_root or output_root not in dataset_dir.parents:
        raise ValueError("--dataset-name must resolve to a subfolder inside --output-dir")

    if args.overwrite and dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    ensure_dir(dataset_dir / "train_pbr")
    ensure_dir(dataset_dir / "meta")
    return output_root, dataset_dir


def validate_append_config(dataset_dir: Path, args: argparse.Namespace) -> None:
    config_path = dataset_dir / "meta" / "generation_config.json"
    if not config_path.exists():
        return
    with config_path.open(encoding="utf-8") as handle:
        previous = json.load(handle)
    for name in ("width", "height", "depth_scale", "model_scale"):
        if name in previous and previous[name] != getattr(args, name):
            raise ValueError(f"Cannot change {name} from {previous[name]} to {getattr(args, name)} "
                             "while appending. Use a new --dataset-name.")


def archive_previous_run_metadata(dataset_dir: Path) -> None:
    """保留上一次运行的配置和报告，根目录仍展示最新一次运行的结果。"""
    meta_dir = dataset_dir / "meta"
    previous = [meta_dir / name for name in ("generation_config.json", "generation_report.json", "quality_report.json")
                if (meta_dir / name).exists()]
    if previous:
        archive_dir = meta_dir / "runs" / f"before_{time.time_ns()}"
        ensure_dir(archive_dir)
        for path in previous:
            shutil.copy2(path, archive_dir / path.name)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    image_dir = Path(args.image_dir).resolve()
    parts_dir = Path(args.parts_dir).resolve()
    
    # 只扫描所选背景目录；未选中的背景资源不会读取或加载。
    hdri_dir = None
    scene_style_dir = None
    hdri_files = []
    scene_styles = []
    if args.background_mode == "hdris":
        hdri_dir = resolve_dir(image_dir, args.hdri_dir, "hdris")
        hdri_files = list_files(hdri_dir, HDRI_SUFFIXES)
        if not hdri_files:
            raise ValueError(f"No .hdr/.exr files found for hdris mode: {hdri_dir}")
        args.disable_walls = True
    else:
        scene_style_dir = resolve_dir(image_dir, args.scene_style_dir, "backgrounds")
        scene_styles = load_scene_styles(scene_style_dir)
        if not scene_styles:
            raise ValueError(f"No background images found for backgrounds mode: {scene_style_dir}")
        if not math.isfinite(args.background_floor_gap) or args.background_floor_gap <= 0.0:
            raise ValueError("--background_floor_gap must be positive and finite")

    ground_texture_dir = resolve_dir(image_dir, args.ground_texture_dir, "ground_texture")
    container_texture_dir = resolve_dir(image_dir, args.container_texture_dir, "container_texture")
    part_texture_dir = resolve_dir(image_dir, args.part_texture_dir, "part_texture")
    ground_textures = list_files(ground_texture_dir, TEXTURE_SUFFIXES)
    if not ground_textures:
        raise ValueError(f"No tabletop texture images found: {ground_texture_dir}")
    output_root, dataset_dir = prepare_output(args)
    validate_append_config(dataset_dir, args)
    first_image_id = resolve_start_image_id(dataset_dir, args.start_image_id)
    get_next_bop_indices(dataset_dir, args.frames_per_chunk)

    bproc.init()
    configure_physics(args)

    container_textures = list_files(container_texture_dir, TEXTURE_SUFFIXES)
    part_textures = list_files(part_texture_dir, TEXTURE_SUFFIXES)

    templates = load_part_templates(parts_dir, args.model_scale, dataset_dir)
    static_scene = create_static_scene(args)
    configure_camera_and_renderer(args)
    write_label_maps(dataset_dir, templates)
    archive_previous_run_metadata(dataset_dir)

    private_path_fields = {
        "parts_dir", "image_dir", "output_dir", "scene_style_dir", "ground_texture_dir",
        "container_texture_dir", "part_texture_dir", "hdri_dir",
    }
    public_config = {k: v for k, v in vars(args).items() if k not in private_path_fields}
    public_config.update({
        "start_image_id_resolved": first_image_id,
        "image_numbering": "continuous_across_chunks",
    })
    save_json(dataset_dir / "meta" / "generation_config.json", public_config)

    print("\n========== DATASET SETUP ==========")
    print(f"Background mode: {args.background_mode}")
    print(f"CAD templates: {len(templates)} from {parts_dir}")
    for item in templates:
        print(f"  Category {item['category_id']}: {item['model_name']} ({item['source_path']})")
    print(f"Scene styles: {len(scene_styles)} from {scene_style_dir}")
    print(f"Ground textures: {len(ground_textures)} from {ground_texture_dir}")
    print(f"Container textures: {len(container_textures)} from {container_texture_dir}")
    print(f"Part textures: {len(part_textures)} from {part_texture_dir}")
    print(f"HDRI files: {len(hdri_files)} from {hdri_dir if hdri_dir is not None else 'disabled'}")
    print(f"Output dataset: {dataset_dir}")
    print(f"First image ID: {first_image_id:06d}; images per folder: {args.frames_per_chunk}")
    print("===================================\n")

    pose_records_path = dataset_dir / "meta" / "pose_records.jsonl"

    stats = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_images": args.num_images,
        "accepted_images": 0,
        "attempts": 0,
        "physics_rejects": 0,
        "depth_rejects": 0,
        "visibility_rejects": 0,
    }
    quality_records = []

    image_id = first_image_id
    for run_index in range(args.num_images):
        accepted = False

        for attempt in range(1, args.max_scene_tries + 1):
            stats["attempts"] += 1
            cleanup_generated_attempt_objects()
            cleanup_unused_materials_and_images()
            reset_keyframes_for_single_frame()
            parts = []
            lights = []
            writing_output = False

            try:
                print(f"[Image {run_index + 1}/{args.num_images}, ID {image_id:06d}] "
                      f"attempt {attempt}/{args.max_scene_tries}")
                
                scene_style_name = randomize_static_scene(
                    static_scene,
                    scene_styles,
                    ground_textures,
                    container_textures,
                    hdri_files,
                    image_id,
                    args,
                )
                
                parts = instantiate_parts(templates, part_textures, args, image_id)
                
                # HDRI 模式下减少额外灯光，防止过曝。
                if hdri_files and random.random() < 0.8:
                    lights = []
                else:
                    lights = create_random_lights(image_id)

                if not simulate_drop(parts, args):
                    stats["physics_rejects"] += 1
                    continue

                cam2world = sample_camera_pose(args)
                bproc.camera.add_camera_pose(cam2world, frame=0)
                assign_pass_indices()

                clear_registered_output("colors")
                render_data = bproc.renderer.render()
                sanitize_render_depths(render_data)
                depth_pixels = valid_depth_pixels(render_data)
                if depth_pixels < args.min_valid_depth_pixels:
                    stats["depth_rejects"] += 1
                    print(f"[Reject] valid depth pixels too low: {depth_pixels}")
                    continue

                visible_count, visible_pixels = visible_instance_stats(render_data, parts)
                strong_visible_count = sum(1 for value in visible_pixels.values() if value >= args.min_visible_pixels)
                if visible_count < args.min_visible_instances or strong_visible_count < args.min_visible_instances:
                    stats["visibility_rejects"] += 1
                    print(f"[Reject] visible instances too low: {visible_pixels}")
                    continue

                chunk_id, frame_id = get_next_bop_indices(dataset_dir, args.frames_per_chunk)
                writing_output = True
                bproc.writer.write_bop(
                    output_dir=str(output_root),
                    target_objects=list(parts),
                    depths=render_data["depth"],
                    colors=render_data["colors"],
                    color_file_format=args.color_format,
                    dataset=args.dataset_name,
                    append_to_existing_output=True,
                    depth_scale=args.depth_scale,
                    jpg_quality=args.jpg_quality,
                    save_world2cam=True,
                    ignore_dist_thres=100.0,
                    annotation_unit="mm",
                    frames_per_chunk=args.frames_per_chunk,
                    calc_mask_info_coco=args.calc_bop_masks and not args.no_bop_masks,
                    delta=0.015,
                    num_worker=args.bop_num_worker,
                )

                write_segmentation_outputs(dataset_dir, chunk_id, frame_id, render_data, parts)
                append_jsonl(
                    pose_records_path,
                    build_pose_record(image_id, chunk_id, frame_id, parts, scene_style_name, visible_pixels),
                )
                quality_records.append(
                    build_quality_record(
                        image_id=image_id,
                        attempt=attempt,
                        chunk_id=chunk_id,
                        frame_id=frame_id,
                        active_parts=parts,
                        scene_style_name=scene_style_name,
                        render_data=render_data,
                        visible_pixels=visible_pixels,
                        depth_pixels=depth_pixels,
                        min_visible_pixels=args.min_visible_pixels,
                    )
                )

                stats["accepted_images"] += 1
                print(
                    f"[OK] progress={run_index + 1}/{args.num_images}, image_id={image_id}, "
                    f"chunk={chunk_id:06d}, frame={frame_id:06d}, "
                    f"visible={visible_pixels}"
                )
                accepted = True
                image_id += 1
                break

            except Exception as exc:
                if writing_output:
                    raise RuntimeError(f"Output failed for image {image_id}; stopped to avoid reusing a "
                                       "partially written image ID") from exc
                print(f"[WARN] attempt failed: {type(exc).__name__}: {exc}")

            finally:
                delete_entities(parts)
                delete_entities(lights)
                cleanup_generated_attempt_objects()
                cleanup_unused_materials_and_images()

        if not accepted:
            print(f"[WARN] Request {run_index + 1} was skipped after {args.max_scene_tries} attempts; "
                  f"image ID {image_id} has not been consumed.")

    stats["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["success"] = stats["accepted_images"] == args.num_images
    save_json(dataset_dir / "meta" / "generation_report.json", stats)
    quality_report_path = dataset_dir / "meta" / "quality_report.json"
    save_json(quality_report_path, build_quality_report(stats, quality_records))

    print("\n========== FINISHED ==========")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"BOP RGB/depth/pose: {dataset_dir / 'train_pbr'}")
    print("BOP official labels: scene_gt.json and scene_camera.json")
    print(f"Segmentation PNGs: {dataset_dir / 'train_pbr' / '000000' / 'seg_category'} and seg_instance")
    print("Visualization PNGs: depth_vis, seg_category_vis, seg_instance_vis")
    print(f"Segmentation labels: {dataset_dir / 'segmentation_labels.json'}")
    print(f"Extra 6D pose records: {pose_records_path}")
    print(f"Quality report: {quality_report_path}")
    print("==============================\n")


if __name__ == "__main__":
    main()
