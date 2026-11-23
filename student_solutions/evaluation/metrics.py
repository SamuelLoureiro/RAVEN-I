"""Shared evaluation metrics for attributes, rules, and problem-solving accuracy."""

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from student_solutions.utils.xml_parser import RULES


def classification_metrics(y_true, y_pred, labels, names=None):
    """Compute accuracy, macro-F1, per-class F1, and confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    names = names or [str(l) for l in labels]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {name: float(val) for name, val in zip(names, per_class)},
        "confusion_matrix": cm,
    }


def evaluate_rules(true_rules_list, pred_rules_list):
    """Evaluate predicted rules against XML ground truth across a list of problems.

    Args:
        true_rules_list: list of prob['rules'] -> [[(rule, attr), ...], ...]
        pred_rules_list: list in the exact same structure returned by a reasoner.
    """
    y_true_rule, y_pred_rule = [], []
    by_attr_true = defaultdict(list)
    by_attr_pred = defaultdict(list)
    attr_matches = []

    for true_comps, pred_comps in zip(true_rules_list, pred_rules_list):
        for t_comp, p_comp in zip(true_comps, pred_comps):
            for (t_rule, t_attr), (p_rule, p_attr) in zip(t_comp, p_comp):
                y_true_rule.append(t_rule)
                y_pred_rule.append(p_rule)
                by_attr_true[t_attr].append(t_rule)
                by_attr_pred[t_attr].append(p_rule)
                attr_matches.append(t_attr == p_attr)

    summary = classification_metrics(y_true_rule, y_pred_rule, labels=RULES, names=RULES)
    summary["attr_match_accuracy"] = float(np.mean(attr_matches)) if attr_matches else 0.0
    summary["by_attribute"] = {}
    for attr in by_attr_true:
        present = sorted(set(by_attr_true[attr]) | set(by_attr_pred[attr]))
        summary["by_attribute"][attr] = {
            "count": len(by_attr_true[attr]),
            "accuracy": float(accuracy_score(by_attr_true[attr], by_attr_pred[attr])),
            "macro_f1": float(f1_score(by_attr_true[attr], by_attr_pred[attr], labels=present, average="macro", zero_division=0)),
        }
    return summary


def evaluate_attributes(y_true_dict, y_pred_dict):
    """Compute accuracy and macro-F1 per attribute, ignoring -1 padded entries."""
    results = {}
    for attr, y_t in y_true_dict.items():
        y_t = np.asarray(y_t).reshape(-1)
        y_p = np.asarray(y_pred_dict[attr]).reshape(-1)
        mask = y_t >= 0
        y_t, y_p = y_t[mask], y_p[mask]
        results[attr] = {
            "accuracy": float(accuracy_score(y_t, y_p)),
            "macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
        }
    return results


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """Render and optionally save a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
