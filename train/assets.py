"""Copy the exact compiled Playground model and only its referenced resources."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

PLAYGROUND_COMMIT = "d59156e099a8785dd58fbce2e25a6ee1f7b5800c"
MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"
MODEL_PATHS = {
    "Go1JoystickFlatTerrain": "assets/go1/walk.xml",
    "Go1Getup": "assets/go1/getup.xml",
    "G1JoystickFlatTerrain": "assets/g1/walk.xml",
}


def export_model(env_name, env, root):
    import mujoco
    import numpy as np
    from mujoco_playground._src import mjx_env

    target = Path(root) / MODEL_PATHS[env_name]
    target.parent.mkdir(parents=True, exist_ok=True)
    # Preserve source numeric tokens: mj_saveLastXML rounds derived geometry.
    def flatten(text):
        element = ET.fromstring(text)
        for parent in list(element.iter()):
            for node in list(parent):
                if node.tag == "include":
                    included = flatten(env._model_assets[Path(node.attrib["file"]).name])
                    position = list(parent).index(node)
                    parent.remove(node)
                    for child in list(included):
                        parent.insert(position, child)
                        position += 1
        return element

    element = flatten(Path(env.xml_path).read_text())
    tree = ET.ElementTree(element)
    for compiler in element.iter("compiler"):
        compiler.set("meshdir", "assets")
        compiler.set("texturedir", "assets")
    option = element.find("option")
    option.set("timestep", str(env.mj_model.opt.timestep))
    option.set("ccd_iterations", str(env.mj_model.opt.ccd_iterations))
    if "Kp" in env._config:
        for position in element.iter("position"):
            position.set("kp", str(env._config.Kp))
        for joint in element.iter("joint"):
            if joint.get("type") != "free":
                joint.set("damping", str(env._config.Kd))
    visual = element.find("visual")
    global_visual = visual.find("global")
    global_visual.set("offwidth", str(env.mj_model.vis.global_.offwidth))
    global_visual.set("offheight", str(env.mj_model.vis.global_.offheight))
    resources = target.parent / "assets"
    resources.mkdir(exist_ok=True)
    for node in tree.iter():
        if "file" not in node.attrib:
            continue
        name = Path(node.attrib["file"]).name
        node.set("file", name)
        payload = env._model_assets[name]
        (resources / name).write_bytes(payload)
    tree.write(target, encoding="unicode")
    restored = mujoco.MjModel.from_xml_path(str(target))
    checks = {}
    for name in dir(env.mj_model):
        if name.startswith("_") or name.endswith("_pathadr"):
            continue  # Allocation sizes and resource-string offsets change when paths move.
        original = getattr(env.mj_model, name)
        if not isinstance(original, np.ndarray) or original.dtype.kind not in "biuf" or not original.size:
            continue
        actual = getattr(restored, name)
        delta = float(np.max(np.abs(actual.astype(float) - original.astype(float))))
        checks[name] = delta
        if not np.array_equal(actual, original):
            raise ValueError(f"Exported {name} differs: {delta}")
    if restored.opt.timestep != env.mj_model.opt.timestep:
        raise ValueError("Export changed timestep")
    robot = "go1" if env_name.startswith("Go1") else "g1"
    shutil.copyfile(Path(mjx_env.MENAGERIE_PATH) / f"unitree_{robot}" / "LICENSE", target.parent / "LICENSE-Menagerie")
    # Playground's Apache license is copied from the pinned checkout by the image.
    shutil.copyfile("/opt/licenses/LICENSE-Playground", target.parent / "LICENSE-Playground")
    metadata = {
        "environment": env_name, "playground_commit": PLAYGROUND_COMMIT,
        "menagerie_commit": MENAGERIE_COMMIT, "mujoco_version": mujoco.__version__,
        "source_xml": str(env.xml_path).split("mujoco_playground/")[-1],
        "export": "Original XML includes flattened, resource paths relocated, runtime options/gains applied; source numeric tokens preserved",
        "model_array_max_abs_errors": checks,
        "comparison_scope": "All nonempty public numeric arrays except resource path-string offsets; paths and private allocation sizes necessarily change with relocation",
        "files": {str(p.relative_to(target.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(target.parent.rglob("*")) if p.is_file() and p.suffix != ".json"},
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return str(target)
