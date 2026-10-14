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
    mujoco.mj_saveLastXML(str(target), env.mj_model)
    tree = ET.parse(target)
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        compiler = ET.SubElement(tree.getroot(), "compiler")
    compiler.set("meshdir", "assets")
    compiler.set("texturedir", "assets")
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
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "geom_size",
                 "geom_friction", "jnt_range", "dof_damping", "dof_armature",
                 "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange"):
        delta = float(np.max(np.abs(getattr(restored, name) - getattr(env.mj_model, name))))
        checks[name] = delta
        if not np.allclose(getattr(restored, name), getattr(env.mj_model, name), atol=2e-6, rtol=2e-6):
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
        "export": "mj_saveLastXML after environment initialization; resource paths relocated",
        "model_array_max_abs_errors": checks,
        "files": {str(p.relative_to(target.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(target.parent.rglob("*")) if p.is_file() and p.suffix != ".json"},
    }
    target.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return str(target)
