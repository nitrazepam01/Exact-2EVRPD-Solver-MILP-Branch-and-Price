"""
Branch-and-Price (B&P) solver for the 2E-VRP-D problem.

Usage:
    python test_cardiff.py [instance1] [instance2] ...

If no arguments given, runs Cardiff10_01 through Cardiff10_10 by default.

Examples:
    python test_cardiff.py                          # Run 01-10
    python test_cardiff.py Cardiff10_01.txt         # Single instance
    python test_cardiff.py Cardiff10_06.txt Cardiff10_07.txt
"""

import sys
import os
import time

# Make sure the script works regardless of working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import CardiffDataLoader
from branch_and_bound import solve_bnp

# Reference values from Table 2 of the paper
PAPER_TABLE2 = {
    "Cardiff10_01.txt": 1223.6,
    "Cardiff10_02.txt": 1905.1,
    "Cardiff10_03.txt": 1412.7,
    "Cardiff10_04.txt": 1633.3,
    "Cardiff10_05.txt": 1682.9,
    "Cardiff10_06.txt": 1221.8,
    "Cardiff10_07.txt": 1931.8,
    "Cardiff10_08.txt": 1578.2,
    "Cardiff10_09.txt": 2552.0,
    "Cardiff10_10.txt": 2045.2,
}

MATCH_TOL = 1.0   # Objective value tolerance for "match"
TIME_LIMIT = 300  # Seconds per instance

def run_instance(inst_file):
    """Run B&P on one instance; return result dict."""
    print(f"\n{'='*60}")
    print(f"Instance: {inst_file}")
    print(f"{'='*60}")

    data = CardiffDataLoader(inst_file)
    print(f"  Customers: {data.n}, Vehicles: {data.k_v}, "
          f"Drones: {data.k_d}, max_drones/route: {data.Gamma}")

    t0 = time.time()
    ub, solution, stats = solve_bnp(data, verbose=True, time_limit=TIME_LIMIT)
    elapsed = time.time() - t0

    paper_ub = PAPER_TABLE2.get(inst_file)
    result = {
        "instance":   inst_file,
        "ub":         ub,
        "paper_ub":   paper_ub,
        "nodes":      stats["nodes"],
        "time":       elapsed,
        "gap":        stats["gap"],
    }

    print(f"\n  ──── Result ────")
    print(f"  Upper Bound : {ub:.4f}")
    if paper_ub is not None:
        diff = abs(ub - paper_ub)
        match = "✓ MATCH" if diff <= MATCH_TOL else "✗ MISMATCH"
        print(f"  Paper   UB  : {paper_ub}   diff={diff:.2f}  {match}")
    print(f"  B&B Nodes   : {stats['nodes']}")
    print(f"  Time        : {elapsed:.2f}s")
    print(f"  Final Gap   : {stats['gap']:.2f}%")

    return result


def main():
    # Determine which instances to run
    if len(sys.argv) > 1:
        instances = sys.argv[1:]
    else:
        instances = [f"Cardiff10_{i:02d}.txt" for i in range(1, 11)]

    results = []
    for inst in instances:
        try:
            r = run_instance(inst)
            results.append(r)
        except FileNotFoundError:
            print(f"  ERROR: {inst} not found.")
        except Exception as e:
            print(f"  ERROR on {inst}: {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'Instance':<20} {'Our UB':>8} {'Paper':>8} {'Diff':>6} {'Nodes':>6} {'Time':>7}"
    print(hdr)
    print("-" * len(hdr))

    matched = 0
    for r in results:
        paper = r["paper_ub"]
        diff_str = f"{abs(r['ub'] - paper):.2f}" if paper else "n/a"
        match_flag = ""
        if paper and abs(r["ub"] - paper) <= MATCH_TOL:
            match_flag = " ✓"
            matched += 1
        elif paper:
            match_flag = " ✗"
        paper_str = f"{paper:.1f}" if paper else "n/a"
        print(f"{r['instance']:<20} {r['ub']:>8.2f} {paper_str:>8} "
              f"{diff_str:>6} {r['nodes']:>6} {r['time']:>6.1f}s{match_flag}")

    print("-" * len(hdr))
    has_paper = sum(1 for r in results if r["paper_ub"])
    if has_paper:
        print(f"Matched: {matched}/{has_paper}  (tolerance = {MATCH_TOL})")


if __name__ == "__main__":
    main()
