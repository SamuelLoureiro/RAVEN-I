"""Parser for RAVEN-I .xml and .npz files across all 7 spatial configurations."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

CONFIGURATIONS = [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
]
RULES = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]
ATTRIBUTES = ["Number", "Position", "Type", "Size", "Color"]

# Lookup tables matching I-RAVEN/const.py (standard shapes have Type in 1..5)
TYPES = ["none", "triangle", "square", "pentagon", "hexagon", "circle", "line"]
SIZES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
COLORS = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]


def decode_meta_matrix(meta_matrix):
    """Decode (12, 9) meta_matrix into [(rule_name, attr_name), ...] per component."""
    rules_by_comp = []
    for c in range(meta_matrix.shape[0] // 4):
        block = meta_matrix[c * 4 : (c + 1) * 4]
        if not block.any():
            continue
        comp_rules = []
        for row in block:
            if not row.any():
                continue
            rule = RULES[int(np.argmax(row[:4]))]
            cols = [i for i, v in enumerate(row[4:9]) if v == 1]
            attr = "Number/Position" if (0 in cols and 1 in cols) else ATTRIBUTES[cols[0]]
            comp_rules.append((rule, attr))
        rules_by_comp.append(comp_rules)
    return rules_by_comp


def decode_answer_mods(meta_answer_mods, num_components=2):
    """Decode (16, 5) or (24, 5) meta_answer_mods into modified (comp_id, attr) per candidate."""
    stride = meta_answer_mods.shape[0] // 8
    mods = []
    for cand in range(8):
        cand_mods = []
        for c in range(num_components):
            row = meta_answer_mods[cand * stride + c]
            for col in range(5):
                if row[col] == 1:
                    cand_mods.append((c, ATTRIBUTES[col]))
        mods.append(cand_mods)
    return mods


def parse_xml(xml_path):
    """Parse panels, rules, and candidate modifications from a RAVEN .xml file."""
    root = ET.parse(xml_path).getroot()
    panels = []
    struct_name = ""

    for panel_el in root.find("Panels").findall("Panel"):
        struct_el = panel_el.find("Struct")
        struct_name = struct_el.get("name", "")
        comps = []
        for comp_el in struct_el.findall("Component"):
            lay_el = comp_el.find("Layout")
            grid = json.loads(lay_el.get("Position"))
            num_slots = len(grid)
            mask = np.zeros(num_slots, dtype=np.uint8)
            entities = []

            for ent_el in lay_el.findall("Entity"):
                bbox = [float(v) for v in json.loads(ent_el.get("bbox"))]
                slot = next(i for i, box in enumerate(grid) if np.allclose(bbox, box, atol=1e-4))
                mask[slot] = 1
                entities.append({
                    "slot": slot,
                    "bbox": bbox,
                    "type": int(ent_el.get("Type")),
                    "size": int(ent_el.get("Size")),
                    "color": int(ent_el.get("Color")),
                    "angle": int(ent_el.get("Angle")),
                })

            number = int(lay_el.get("Number"))
            comps.append({
                "component_id": int(comp_el.get("id")),
                "name": comp_el.get("name", ""),
                "layout": lay_el.get("name", ""),
                "number": number,
                "count": number + 1,
                "uniformity": int(lay_el.get("Uniformity")) == 3,
                "num_slots": num_slots,
                "positions": sorted(e["slot"] for e in entities),
                "position_mask": mask,
                "entities": entities,
                "type": entities[0]["type"],
                "size": entities[0]["size"],
                "color": entities[0]["color"],
            })
        panels.append(comps)

    rules = []
    for rg in sorted(root.find("Rules").findall("Rule_Group"), key=lambda e: int(e.get("id"))):
        rules.append([(r.get("name"), r.get("attr")) for r in rg.findall("Rule")])

    candidate_mods = []
    for cand in sorted(root.find("Modified_attributes").findall("Candidate"), key=lambda e: int(e.get("id"))):
        candidate_mods.append([(int(a.get("component_id")), a.get("name")) for a in cand.findall("Attribute")])

    return {
        "struct": struct_name,
        "panels": panels,
        "rules": rules,
        "candidate_mods": candidate_mods,
    }


def load_problem(npz_path, load_images=True):
    """Load one RAVEN problem from its .npz and matching .xml files."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as d:
        images = d["image"].copy() if load_images else None
        target = int(d["target"])
        meta_matrix = d["meta_matrix"].copy()
        meta_target = d["meta_target"].copy()
        meta_structure = d["meta_structure"].copy()
        meta_answer_mods = d["meta_answer_mods"].copy()

    xml = parse_xml(npz_path.with_suffix(".xml"))
    return {
        "npz_path": str(npz_path),
        "config": npz_path.parent.name,
        "split": npz_path.stem.rsplit("_", 1)[-1],
        "struct": xml["struct"],
        "num_components": len(xml["panels"][0]),
        "target": target,
        "images": images,
        "panels": xml["panels"],
        "rules": xml["rules"],
        "candidate_mods": xml["candidate_mods"],
        "meta_matrix": meta_matrix,
        "meta_target": meta_target,
        "meta_structure": meta_structure,
        "meta_answer_mods": meta_answer_mods,
    }
