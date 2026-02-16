#!/usr/bin/env python3
"""
Fusion Threshold Tuning Script - Sequential Optimization

Tunes fusion thresholds one at a time to find optimal values:
1. Unanimous confidence boost
2. Error-aware down-weighting factor
3. Low confidence threshold for routing

Uses sequential optimization (tune one, fix it, tune next) for speed.
Runs on 20-sample validation set.
"""
import json
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Sequential tuning grids
UNANIMOUS_BOOST_GRID = [1.0, 1.1, 1.2, 1.3, 1.5]
DOWN_WEIGHT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
LOW_CONF_GRID = [0.5, 0.6, 0.7, 0.8]

# Defaults (current values)
DEFAULT_BOOST = 1.2
DEFAULT_DOWN_WEIGHT = 0.5
DEFAULT_LOW_CONF = 0.7

NUM_SAMPLES = 20


def patch_ensemble_helpers(boost: float, down_weight: float, low_conf: float, backup_path: Path):
    """Temporarily patch ensemble_helpers.py with new threshold values."""
    helpers_file = Path("src/utils/ensemble_helpers.py")

    # Backup on first call
    if not backup_path.exists():
        backup_path.write_text(helpers_file.read_text())

    # Read original
    content = backup_path.read_text()

    # Patch unanimous boost (line ~341)
    content = content.replace(
        "min(1.0, float(np.mean(calibrated_confidences)) * 1.2),",
        f"min(1.0, float(np.mean(calibrated_confidences)) * {boost}),"
    )

    # Patch down-weight factor (lines ~389, 393, 397)
    content = content.replace(
        "weights['llama-3.2-3b'] *= 0.5",
        f"weights['llama-3.2-3b'] *= {down_weight}"
    )
    content = content.replace(
        "weights['mistral-7b'] *= 0.5",
        f"weights['mistral-7b'] *= {down_weight}"
    )
    content = content.replace(
        "weights['gemma-1.1-7b'] *= 0.5",
        f"weights['gemma-1.1-7b'] *= {down_weight}"
    )

    # Patch low conf threshold (line ~396)
    content = content.replace(
        "model_results['gemma-1.1-7b'].get('confidence', 0.0) < 0.7:",
        f"model_results['gemma-1.1-7b'].get('confidence', 0.0) < {low_conf}:"
    )

    helpers_file.write_text(content)


def restore_original(backup_path: Path):
    """Restore original ensemble_helpers.py."""
    helpers_file = Path("src/utils/ensemble_helpers.py")
    helpers_file.write_text(backup_path.read_text())


def run_test(boost: float, down_weight: float, low_conf: float) -> Tuple[float, float]:
    """Run test and return (EM, F1)."""
    cmd = [
        "python", "main.py",
        "--dataset", "drop",
        "--samples", str(NUM_SAMPLES),
        "--no-output-capture"
    ]

    try:
        result = subprocess.run(
            ["conda", "run", "-n", "hysym"] + cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env={"CONDA_NO_PLUGINS": "true", **subprocess.os.environ},
            cwd=Path.cwd()
        )

        output = result.stdout
        em_line = [l for l in output.split('\n') if 'Avg EM:' in l]
        f1_line = [l for l in output.split('\n') if 'Avg F1:' in l]

        if em_line and f1_line:
            em = float(em_line[0].split(':')[1].strip())
            f1 = float(f1_line[0].split(':')[1].strip())
            return (em, f1)
        else:
            print(f"  WARNING: Could not parse metrics")
            return (0.0, 0.0)

    except subprocess.TimeoutExpired:
        print(f"  WARNING: Timeout")
        return (0.0, 0.0)
    except Exception as e:
        print(f"  WARNING: Error - {e}")
        return (0.0, 0.0)


def tune_parameter(param_name: str, param_grid: List[float],
                   fixed_boost: float, fixed_down: float, fixed_low: float) -> Tuple[float, float, float]:
    """
    Tune a single parameter while keeping others fixed.
    Returns (best_value, best_em, best_f1).
    """
    print(f"\n{'='*60}")
    print(f"Tuning: {param_name}")
    print(f"{'='*60}")
    print(f"Grid: {param_grid}")
    print(f"Fixed values: boost={fixed_boost}, down={fixed_down}, low_conf={fixed_low}\n")

    results = []
    for value in param_grid:
        # Set parameter value
        if param_name == "unanimous_boost":
            test_boost, test_down, test_low = value, fixed_down, fixed_low
        elif param_name == "down_weight":
            test_boost, test_down, test_low = fixed_boost, value, fixed_low
        else:  # low_conf
            test_boost, test_down, test_low = fixed_boost, fixed_down, value

        print(f"Testing {param_name}={value}... ", end='', flush=True)
        em, f1 = run_test(test_boost, test_down, test_low)
        print(f"EM={em:.3f}, F1={f1:.3f}")

        results.append((value, em, f1))

    # Find best
    best = max(results, key=lambda x: x[1])  # Max EM
    print(f"\nBest {param_name}: {best[0]} (EM={best[1]:.3f}, F1={best[2]:.3f})")

    return best


def main():
    print("=" * 80)
    print("FUSION THRESHOLD TUNING - Sequential Optimization")
    print("=" * 80)
    print(f"Validation samples: {NUM_SAMPLES}")
    print(f"Total tests: {len(UNANIMOUS_BOOST_GRID) + len(DOWN_WEIGHT_GRID) + len(LOW_CONF_GRID)}")
    print(f"Estimated time: ~{(len(UNANIMOUS_BOOST_GRID) + len(DOWN_WEIGHT_GRID) + len(LOW_CONF_GRID)) * 4} minutes\n")

    # Create backup
    backup_path = Path(tempfile.mktemp(suffix=".ensemble_helpers.bak"))

    try:
        # Stage 1: Tune unanimous boost
        best_boost, em1, f1_1 = tune_parameter(
            "unanimous_boost", UNANIMOUS_BOOST_GRID,
            DEFAULT_BOOST, DEFAULT_DOWN_WEIGHT, DEFAULT_LOW_CONF
        )
        patch_ensemble_helpers(best_boost, DEFAULT_DOWN_WEIGHT, DEFAULT_LOW_CONF, backup_path)

        # Stage 2: Tune down-weight (with best boost)
        best_down, em2, f1_2 = tune_parameter(
            "down_weight", DOWN_WEIGHT_GRID,
            best_boost, DEFAULT_DOWN_WEIGHT, DEFAULT_LOW_CONF
        )
        patch_ensemble_helpers(best_boost, best_down, DEFAULT_LOW_CONF, backup_path)

        # Stage 3: Tune low conf threshold (with best boost + down-weight)
        best_low, em3, f1_3 = tune_parameter(
            "low_conf", LOW_CONF_GRID,
            best_boost, best_down, DEFAULT_LOW_CONF
        )

        # Final configuration
        print("\n" + "=" * 80)
        print("OPTIMAL CONFIGURATION FOUND")
        print("=" * 80)
        print(f"\nunanimous_boost: {best_boost} (default: {DEFAULT_BOOST})")
        print(f"down_weight_factor: {best_down} (default: {DEFAULT_DOWN_WEIGHT})")
        print(f"low_conf_threshold: {best_low} (default: {DEFAULT_LOW_CONF})")

        # Test final configuration
        print(f"\nTesting final configuration...")
        patch_ensemble_helpers(best_boost, best_down, best_low, backup_path)
        final_em, final_f1 = run_test(best_boost, best_down, best_low)

        print(f"\nFinal performance:")
        print(f"  EM: {final_em:.3f}")
        print(f"  F1: {final_f1:.3f}")

        # Save results
        results = {
            'optimal_config': {
                'unanimous_boost': best_boost,
                'down_weight_factor': best_down,
                'low_conf_threshold': best_low
            },
            'default_config': {
                'unanimous_boost': DEFAULT_BOOST,
                'down_weight_factor': DEFAULT_DOWN_WEIGHT,
                'low_conf_threshold': DEFAULT_LOW_CONF
            },
            'final_metrics': {
                'em': final_em,
                'f1': final_f1
            }
        }

        results_file = Path("fusion_threshold_tuning_results.json")
        results_file.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {results_file}")

    finally:
        # Restore original
        if backup_path.exists():
            restore_original(backup_path)
            backup_path.unlink()

    return 0


if __name__ == "__main__":
    sys.exit(main())
