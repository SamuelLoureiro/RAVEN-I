"""PyTorch Datasets for full RAVEN-I problems and individual panels."""

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from student_solutions.utils.xml_parser import (
    CONFIGURATIONS,
    RULES,
    load_problem,
    parse_xml,
)

MAX_COMPONENTS = 2
MAX_SLOTS = 9


def get_files(data_dir, configs=None, split="train", max_per_config=None):
    """Return sorted (config_name, npz_path) pairs for a given split ('train'/'val'/'test')."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")
    root = Path(data_dir)
    if configs is None:
        cfg_list = [c for c in CONFIGURATIONS if (root / c).is_dir()]
    elif isinstance(configs, str):
        cfg_list = [configs]
    else:
        cfg_list = list(configs)

    files = []
    for cfg in cfg_list:
        paths = sorted(
            (root / cfg).glob(f"*_{split}.npz"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if max_per_config is not None:
            paths = paths[:max_per_config]
        files.extend((cfg, p) for p in paths)
    if not files:
        raise FileNotFoundError(f"No *_{split}.npz files found in {root}")
    return files


def _pack_panel(panel_comps):
    """Pack up to 2 components of one panel into fixed-size padded arrays."""
    t = np.full(MAX_COMPONENTS, -1, dtype=np.int64)
    s = np.full(MAX_COMPONENTS, -1, dtype=np.int64)
    c = np.full(MAX_COMPONENTS, -1, dtype=np.int64)
    n = np.full(MAX_COMPONENTS, -1, dtype=np.int64)
    pos = np.zeros((MAX_COMPONENTS, MAX_SLOTS), dtype=np.float32)

    for i, comp in enumerate(panel_comps[:MAX_COMPONENTS]):
        t[i] = comp["type"] - 1  # XML 1..5 -> 0..4 for CrossEntropyLoss
        s[i] = comp["size"]
        c[i] = comp["color"]
        n[i] = comp["number"]
        pos[i, : comp["num_slots"]] = comp["position_mask"]
    return t, s, c, n, pos


def _pack_rules(rules_by_comp):
    """Pack rule indices (0..3) into shape (2, 4) for [Number/Position, Type, Size, Color]."""
    out = np.full((MAX_COMPONENTS, 4), -1, dtype=np.int64)
    slot_map = {"Number/Position": 0, "Number": 0, "Position": 0, "Type": 1, "Size": 2, "Color": 3}
    for i, comp_rules in enumerate(rules_by_comp[:MAX_COMPONENTS]):
        for rule_name, attr_name in comp_rules:
            out[i, slot_map[attr_name]] = RULES.index(rule_name)
    return out


def _resize_batch(images, image_size):
    """Resize (N, H, W) grayscale images if image_size is given."""
    if image_size is None or images.shape[-1] == image_size:
        return images
    return np.stack([cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA) for img in images])


class RAVENDataset(Dataset):
    """Problem-level dataset: returns all 16 images, attributes, rules, and target (0..7)."""

    def __init__(self, data_dir, configs=None, split="train", image_size=None, max_per_config=None):
        self.samples = get_files(data_dir, configs, split, max_per_config)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def get_problem(self, idx):
        """Return full parsed symbolic dict + images for visualization or symbolic solving."""
        return load_problem(self.samples[idx][1], load_images=True)

    def __getitem__(self, idx):
        cfg, npz_path = self.samples[idx]
        prob = load_problem(npz_path, load_images=True)
        imgs = _resize_batch(prob["images"], self.image_size).astype(np.float32) / 255.0

        t = np.full((16, MAX_COMPONENTS), -1, dtype=np.int64)
        s = np.full((16, MAX_COMPONENTS), -1, dtype=np.int64)
        c = np.full((16, MAX_COMPONENTS), -1, dtype=np.int64)
        n = np.full((16, MAX_COMPONENTS), -1, dtype=np.int64)
        pos = np.zeros((16, MAX_COMPONENTS, MAX_SLOTS), dtype=np.float32)
        for p_idx, comps in enumerate(prob["panels"]):
            t[p_idx], s[p_idx], c[p_idx], n[p_idx], pos[p_idx] = _pack_panel(comps)

        return {
            "images": torch.from_numpy(imgs),
            "target": torch.tensor(prob["target"], dtype=torch.long),
            "type": torch.from_numpy(t),
            "size": torch.from_numpy(s),
            "color": torch.from_numpy(c),
            "number": torch.from_numpy(n),
            "position_mask": torch.from_numpy(pos),
            "rules": torch.from_numpy(_pack_rules(prob["rules"])),
            "num_components": prob["num_components"],
            "config_idx": CONFIGURATIONS.index(cfg),
        }


class RAVENPanelDataset(Dataset):
    """Panel-level dataset: returns one (1, H, W) image and its attributes for CNN training."""

    def __init__(self, data_dir, configs="center_single", split="train", image_size=None, max_per_config=None):
        self.samples = get_files(data_dir, configs, split, max_per_config)
        self.image_size = image_size
        n_prob = len(self.samples)

        # Parse all XML labels once here so training epochs don't re-parse XML files
        self.type = np.full((n_prob, 16, MAX_COMPONENTS), -1, dtype=np.int64)
        self.size = np.full((n_prob, 16, MAX_COMPONENTS), -1, dtype=np.int64)
        self.color = np.full((n_prob, 16, MAX_COMPONENTS), -1, dtype=np.int64)
        self.number = np.full((n_prob, 16, MAX_COMPONENTS), -1, dtype=np.int64)
        self.position_mask = np.zeros((n_prob, 16, MAX_COMPONENTS, MAX_SLOTS), dtype=np.float32)
        self.num_components = np.zeros(n_prob, dtype=np.int64)
        self.config_idx = np.zeros(n_prob, dtype=np.int64)

        for i, (cfg, npz_path) in enumerate(self.samples):
            xml = parse_xml(npz_path.with_suffix(".xml"))
            self.num_components[i] = len(xml["panels"][0])
            self.config_idx[i] = CONFIGURATIONS.index(cfg)
            for p, comps in enumerate(xml["panels"]):
                self.type[i, p], self.size[i, p], self.color[i, p], self.number[i, p], self.position_mask[i, p] = _pack_panel(comps)

    def __len__(self):
        return len(self.samples) * 16

    def __getitem__(self, idx):
        prob_idx, panel_idx = divmod(idx, 16)
        with np.load(self.samples[prob_idx][1], allow_pickle=False) as d:
            img = d["image"][panel_idx]
        if self.image_size is not None and img.shape[-1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        return {
            "image": torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0),
            "type": torch.from_numpy(self.type[prob_idx, panel_idx]),
            "size": torch.from_numpy(self.size[prob_idx, panel_idx]),
            "color": torch.from_numpy(self.color[prob_idx, panel_idx]),
            "number": torch.from_numpy(self.number[prob_idx, panel_idx]),
            "position_mask": torch.from_numpy(self.position_mask[prob_idx, panel_idx]),
            "num_components": int(self.num_components[prob_idx]),
            "config_idx": int(self.config_idx[prob_idx]),
        }
