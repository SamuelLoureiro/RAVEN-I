"""Evaluate the symbolic solver on ground-truth XML attributes (Oracle baseline)."""

import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score

from student_solutions.evaluation.metrics import evaluate_rules
from student_solutions.models.symbolic_solver import solve_problem
from student_solutions.utils.data_loader import get_files
from student_solutions.utils.xml_parser import CONFIGURATIONS, RULES, load_problem


def main():
    parser = argparse.ArgumentParser(description="Evaluate symbolic solver on XML oracle attributes.")
    parser.add_argument("--data-dir", type=str, default="data/test-1k")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--configs", nargs="+", default=CONFIGURATIONS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_true_rules, all_pred_rules = [], []
    all_true_ans, all_pred_ans = [], []

    print(f"=== Symbolic Oracle Evaluation (split={args.split}, dir={data_dir}) ===")
    print(f"{'Configuration':<42} {'N':>4}  {'Rule Acc':>9}  {'Rule F1':>8}  {'Solve Acc':>9}")
    print("-" * 78)

    for cfg in args.configs:
        true_rules, pred_rules = [], []
        true_ans, pred_ans = [], []
        for _, npz_path in get_files(data_dir, cfg, args.split):
            prob = load_problem(npz_path, load_images=False)
            res = solve_problem(prob["panels"])
            true_rules.append(prob["rules"])
            pred_rules.append(res["rules"])
            true_ans.append(prob["target"])
            pred_ans.append(res["answer"])

        r_metrics = evaluate_rules(true_rules, pred_rules)
        solve_acc = accuracy_score(true_ans, pred_ans)
        print(
            f"{cfg:<42} {len(true_ans):>4}  {r_metrics['accuracy']*100:>8.2f}%  "
            f"{r_metrics['macro_f1']*100:>7.2f}%  {solve_acc*100:>8.2f}%"
        )

        all_true_rules.extend(true_rules)
        all_pred_rules.extend(pred_rules)
        all_true_ans.extend(true_ans)
        all_pred_ans.extend(pred_ans)

    overall_rules = evaluate_rules(all_true_rules, all_pred_rules)
    overall_solve = accuracy_score(all_true_ans, all_pred_ans)
    print("-" * 78)
    print(
        f"{'OVERALL':<42} {len(all_true_ans):>4}  {overall_rules['accuracy']*100:>8.2f}%  "
        f"{overall_rules['macro_f1']*100:>7.2f}%  {overall_solve*100:>8.2f}%"
    )

    print("\n--- Per-Rule F1-Score ---")
    for r_name, f1_val in overall_rules["per_class_f1"].items():
        print(f"  {r_name:<18}: {f1_val*100:.2f}%")

    print(f"\n--- Per-Attribute Rule Accuracy (Attr disambiguation acc: {overall_rules['attr_match_accuracy']*100:.2f}%) ---")
    for attr_name, stats in overall_rules["by_attribute"].items():
        print(f"  {attr_name:<18} (n={stats['count']:>4}): Acc={stats['accuracy']*100:.2f}%, F1={stats['macro_f1']*100:.2f}%")

    print("\n--- Rule Confusion Matrix (rows=True, cols=Pred) ---")
    print("  " + "  ".join(f"{r[:6]:>6}" for r in RULES))
    for r_name, row in zip(RULES, overall_rules["confusion_matrix"]):
        print(f"  " + "  ".join(f"{val:>6d}" for val in row) + f"  | {r_name}")


if __name__ == "__main__":
    main()
