"""Restore Menagerie body-floor collision shapes without changing G1 dynamics.

Playground's walking XML retains Menagerie's collision meshes/cylinders as
zero-density visual copies. The falling scene reuses those copies and the
Playground limb capsules; only geometry groups/names and floor pairs change.
Source: unitree_g1/g1.xml at the Menagerie revision in train.assets.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

from train.assets import MENAGERIE_COMMIT, PLAYGROUND_COMMIT


def build_fallen(root):
    root = Path(root)
    source = root / "assets/g1/walk.xml"
    tree = ET.parse(source)
    element = tree.getroot()
    contact = element.find("contact")
    enabled = []
    for body in element.iter("body"):
        seen = set()
        for geom in body.findall("geom"):
            mesh = geom.get("mesh")
            # Menagerie collision shapes were retained as duplicate visual
            # meshes, plus one cylinder at each shoulder joint.
            restored = (mesh is not None and mesh in seen) or geom.get("type") == "cylinder"
            if mesh:
                seen.add(mesh)
            if restored:
                name = f"fallen_{body.get('name')}_{mesh or 'cylinder'}"
                geom.set("name", name)
                geom.set("group", "3")
                # Keep class=visual: it supplies density=0 and mesh type.
                # Explicit floor pairs enable contact despite the zero masks.
                enabled.append(name)
            elif geom.get("class") == "collision":
                enabled.append(geom.attrib["name"])
    for name in enabled:
        ET.SubElement(contact, "pair", name=f"{name}_floor", geom1=name,
                      geom2="floor", condim="3", friction="0.6 0.6")
    destination = source.with_name("fallen.xml")
    tree.write(destination, encoding="unicode")
    metadata = {
        "source_model": "assets/g1/walk.xml",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "playground_commit": PLAYGROUND_COMMIT,
        "menagerie_commit": MENAGERIE_COMMIT,
        "collision_source": f"https://github.com/google-deepmind/mujoco_menagerie/blob/{MENAGERIE_COMMIT}/unitree_g1/g1.xml",
        "generator": "python -m train.fallen",
        "changes": "Reuse existing zero-density Menagerie mesh/cylinder copies and Playground capsules for explicit body-floor pairs. No new self-collision pairs. Robot kinematics, inertial data, joints, actuators and simulation options unchanged.",
        "floor_collision_geoms": enabled,
        "purpose": "Browser fall handling and the separately trained experimental G1 get-up task; not the G1 walking training/evaluation model.",
    }
    destination.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    build_fallen(Path(__file__).resolve().parents[1])
