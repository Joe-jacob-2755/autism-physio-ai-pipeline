"""
=============================================================================
MODULE 4 – DATA ANALYSER  |  main.py
=============================================================================
Interactive CLI entry point for Module 4.

Launch:
  run_data_analyser             (from repo root, Windows)
  ./run_data_analyser.sh        (Mac/Linux)
  python main.py                (direct)
  python main.py --source <M3_folder> --threshold_window 300
=============================================================================
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT  = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from analyser import DataAnalyser
from config   import MODULE_VERSION, DEFAULT_THRESHOLD_WINDOW_S, \
                     DEFAULT_THRESHOLD_MAD_FACTOR, DEFAULT_THRESHOLD_SUSTAIN_S

WIDTH = 64

def _clear(): os.system("cls" if os.name == "nt" else "clear")
def _section(t): print(f"\n  {'─'*(WIDTH-4)}\n  {t}\n  {'─'*(WIDTH-4)}")
def _ask(p, d=""): return (input(f"\n  {p} [{d}]: ").strip() or str(d))
def _ask_float(p, d, lo=0.0):
    while True:
        try:
            v = float(_ask(p, d))
            if v >= lo: return v
            print(f"  ✗ Must be ≥ {lo}")
        except ValueError:
            print("  ✗ Please enter a number")
def _ask_yn(p, d=True):
    raw = input(f"\n  {p}  ({'Y/n' if d else 'y/N'}): ").strip().lower()
    return d if raw == "" else raw in ("y","yes")


def _browse_source() -> Path:
    _section("📂  Select Module 3 output folder to analyse")
    candidates, labels = [], []
    for root, tag in [
        (REPO_ROOT / "module_3_preprocessing" / "outputs", "M3"),
        (REPO_ROOT / "module_2a_data_simulation" / "outputs", "M2A"),
        (REPO_ROOT / "module_1_data_acquisition" / "outputs", "M1"),
    ]:
        if root.exists():
            for d in sorted(root.iterdir()):
                if not d.is_dir(): continue
                n_csv = len(list(d.rglob("*.csv")))
                if n_csv == 0: continue
                try: rel = d.relative_to(REPO_ROOT)
                except: rel = d
                candidates.append(d)
                labels.append(f"[{tag}]  {rel.name}  ({n_csv} CSVs)")
    labels.append("📁  Type a custom path")
    print()
    for i, l in enumerate(labels[:-1], 1):
        print(f"    {i}.  {l}")
    print(f"    {len(candidates)+1}.  {labels[-1]}")
    while True:
        raw = input("\n  Select [1]: ").strip() or "1"
        try:
            idx = int(raw) - 1
            if idx == len(candidates):
                p = Path(input("  Enter path: ").strip())
                return p if p.exists() else None
            if 0 <= idx < len(candidates):
                return candidates[idx]
        except ValueError:
            p = Path(raw)
            if p.exists(): return p
        print(f"  ✗  Enter 1–{len(candidates)+1}.")


def _interactive():
    _clear()
    print()
    print("  " + "═" * (WIDTH - 2))
    print("  ║    🧠  AUTISM PHYSIO-AI PIPELINE" + " "*(WIDTH-38) + "║")
    print("  ║        Module 4 — Data Analyser" + " "*(WIDTH-36) + "║")
    print(f"  ║        v{MODULE_VERSION}" + " "*(WIDTH-14) + "║")
    print("  " + "═" * (WIDTH - 2))

    source = _browse_source()
    if source is None:
        print("  ✗  No valid source selected.")
        return

    _section("⚙️  Adaptive threshold settings")
    print("""
  The adaptive threshold flags periods where a signal deviates more than
  N × MAD from its running median for longer than a set duration.
  Good practice values: window=300s, N=3, sustain=30s.
""")
    win_s    = _ask_float("Threshold window (seconds)", DEFAULT_THRESHOLD_WINDOW_S, 30)
    mad_n    = _ask_float("Deviation multiplier (N × MAD)", DEFAULT_THRESHOLD_MAD_FACTOR, 1)
    sustain  = _ask_float("Minimum sustained duration (seconds)", DEFAULT_THRESHOLD_SUSTAIN_S, 1)

    session_id = _ask("Session ID", source.name)
    user_id    = _ask("User ID", "user_001")

    print(f"""
  ✅  Settings:
    Source    : {source}
    Window    : {win_s:.0f}s  |  N×MAD: {mad_n}  |  Sustain: {sustain:.0f}s
    Session   : {session_id}  |  User: {user_id}
""")
    if not _ask_yn("Start analysis?", True):
        print("  Cancelled.")
        return

    DataAnalyser(
        threshold_window_s   = win_s,
        threshold_mad_factor = mad_n,
        threshold_sustain_s  = sustain,
        verbose              = True,
    ).run(source=source, session_id=session_id, user_id=user_id)


def _cli():
    p = argparse.ArgumentParser(description=f"Module 4 Data Analyser v{MODULE_VERSION}")
    p.add_argument("--source", required=True, help="Module 3 output folder path")
    p.add_argument("--threshold_window",  type=float, default=DEFAULT_THRESHOLD_WINDOW_S)
    p.add_argument("--threshold_mad",     type=float, default=DEFAULT_THRESHOLD_MAD_FACTOR)
    p.add_argument("--threshold_sustain", type=float, default=DEFAULT_THRESHOLD_SUSTAIN_S)
    p.add_argument("--session_id", default="cli_session")
    p.add_argument("--user_id",    default="user_001")
    args = p.parse_args()
    DataAnalyser(
        threshold_window_s   = args.threshold_window,
        threshold_mad_factor = args.threshold_mad,
        threshold_sustain_s  = args.threshold_sustain,
        verbose              = True,
    ).run(source=Path(args.source), session_id=args.session_id, user_id=args.user_id)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _interactive()
    else:
        _cli()
