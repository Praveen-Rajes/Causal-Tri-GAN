#!/usr/bin/env python3
"""
CausalTriGAN-ProjectedGAN - Ablation Runner
Runs all ablation experiments sequentially.

"""
import os
import sys
import json
import argparse
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import ABLATION_CONFIGS


ALL_ABLATIONS = [
    "no_causal",
    "no_anat",
    "no_percep",
    "no_div",
    "no_diffaugment",
    "no_progressive",
    "causal_nec_only",
    "causal_suf_only",
]


def run_ablation(ablation_name, args):
    """Run a single ablation training + evaluation."""
    print(f"\n{'#'*70}")
    print(f"  ABLATION: {ablation_name} (ProjectedGAN)")
    print(f"{'#'*70}\n")

    start = time.time()

    train_cmd = [
        sys.executable, "scripts/train.py",
        "--ablation", ablation_name,
    ]
    if args.data_root:
        train_cmd.extend(["--data_root", args.data_root])
    if args.kimg:
        train_cmd.extend(["--kimg", str(args.kimg)])
    if args.batch_size:
        train_cmd.extend(["--batch_size", str(args.batch_size)])

    log_dir = "outputs/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ablation_{ablation_name}.log")

    print(f"  Training log: {log_file}")
    print(f"  Command: {' '.join(train_cmd)}")

    with open(log_file, 'w') as f:
        result = subprocess.run(train_cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  [ERROR] Training failed for {ablation_name}! Check {log_file}")
        return False

    train_time = time.time() - start
    print(f"  Training complete in {train_time/3600:.1f} hours")

    if not args.skip_eval:
        ckpt_dir = "outputs/checkpoints"
        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_1200kimg.pt")
        if not os.path.exists(ckpt_path):
            ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')])
            if ckpts:
                ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
            else:
                print(f"  [ERROR] No checkpoint found for {ablation_name}")
                return False

        eval_cmd = [
            sys.executable, "scripts/evaluate.py",
            "--checkpoint", ckpt_path,
            "--ablation", ablation_name,
        ]
        if args.data_root:
            eval_cmd.extend(["--data_root", args.data_root])

        eval_log = os.path.join(log_dir, f"eval_{ablation_name}.log")
        print(f"  Evaluation log: {eval_log}")

        with open(eval_log, 'w') as f:
            result = subprocess.run(eval_cmd, stdout=f, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"  [WARNING] Evaluation failed for {ablation_name}")

    total_time = time.time() - start
    print(f"  Total time for {ablation_name}: {total_time/3600:.1f} hours")
    return True


def compile_results():
    """Compile all ablation results into a summary table."""
    eval_dir = "outputs/eval_results"
    results = {}

    for f in os.listdir(eval_dir):
        if f.startswith("eval_results_") and f.endswith(".json"):
            ablation = f.replace("eval_results_", "").replace(".json", "")
            with open(os.path.join(eval_dir, f)) as fh:
                results[ablation] = json.load(fh)

    if not results:
        print("No results found to compile.")
        return

    print(f"\n{'='*90}")
    print(f"  ABLATION RESULTS SUMMARY (ProjectedGAN)")
    print(f"{'='*90}")
    header = f"{'Variant':<20} {'FID':>8} {'Oracle AUC':>12} {'Sufficiency':>12} {'Necessity':>12} {'GC-IoU':>8}"
    print(header)
    print("-" * 90)

    for name in ["full"] + ALL_ABLATIONS:
        if name in results:
            r = results[name]
            fid = r.get("FID", "N/A")
            auc = r.get("oracle_auc_mean", "N/A")
            suf = r.get("sufficiency_score", "N/A")
            nec = r.get("necessity_score", "N/A")
            iou = r.get("gradcam_iou", "N/A")

            fid_s = f"{fid:.2f}" if isinstance(fid, float) else fid
            auc_s = f"{auc:.4f}" if isinstance(auc, float) else auc
            suf_s = f"{suf:.4f}" if isinstance(suf, float) else suf
            nec_s = f"{nec:.4f}" if isinstance(nec, float) else nec
            iou_s = f"{iou:.4f}" if isinstance(iou, float) else iou

            print(f"{name:<20} {fid_s:>8} {auc_s:>12} {suf_s:>12} {nec_s:>12} {iou_s:>8}")

    print(f"{'='*90}")

    compiled_path = os.path.join(eval_dir, "ablation_summary.json")
    with open(compiled_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nCompiled results saved to: {compiled_path}")


def main():
    parser = argparse.ArgumentParser(description="CausalTriGAN-ProjectedGAN Ablation Runner")
    parser.add_argument("--ablations", nargs="+", default=None,
                        help=f"Ablations to run (default: all). Options: {ALL_ABLATIONS}")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--compile_only", action="store_true")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--kimg", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.compile_only:
        compile_results()
        return

    ablations = args.ablations or ALL_ABLATIONS
    print(f"Running {len(ablations)} ablation(s): {ablations}")

    start_total = time.time()
    results_summary = {}

    for abl in ablations:
        if abl not in ABLATION_CONFIGS:
            print(f"[WARNING] Unknown ablation '{abl}', skipping.")
            continue
        success = run_ablation(abl, args)
        results_summary[abl] = "SUCCESS" if success else "FAILED"

    total_time = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"  ALL ABLATIONS COMPLETE (ProjectedGAN)")
    print(f"  Total time: {total_time/3600:.1f} hours")
    for name, status in results_summary.items():
        print(f"    {name}: {status}")
    print(f"{'='*60}")

    compile_results()


if __name__ == "__main__":
    main()
