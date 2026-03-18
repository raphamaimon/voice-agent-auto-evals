"""Results viewer for completed runs.

Usage: python main.py results
"""

from __future__ import annotations

import json
import os

from .config import Config
from . import display


def show_results(cfg: Config) -> None:
    """Print a summary of the last completed run."""

    out_dir = cfg.output.dir
    log_path = os.path.join(out_dir, "autoresearch.json")
    tsv_path = os.path.join(out_dir, "results.tsv")
    best_path = os.path.join(out_dir, "best_prompt.txt")

    if not os.path.exists(log_path):
        print(f"No results found in {out_dir}/")
        print("Run 'python main.py research' first.")
        return

    with open(log_path) as f:
        data = json.load(f)

    meta = data.get("meta", {})
    experiments = data.get("experiments", [])
    original = data.get("original_prompt", "")
    best = data.get("best_prompt", "")
    eval_suite = data.get("eval_suite", [])

    # --- Header ---
    display.header("AutoVoiceEvals — Results")

    display.info(f"Assistant:    {meta.get('assistant', '?')}")
    display.info(f"LLM:          {meta.get('llm', '?')}")
    display.info(f"Started:      {meta.get('started', '?')}")
    display.info(f"Ended:        {meta.get('ended', '?')}")
    display.info(f"Experiments:  {meta.get('total_experiments', len(experiments))}")

    # --- Eval suite ---
    display.section("EVAL SUITE")
    for s in eval_suite:
        d = s.get("difficulty", "?")
        name = s.get("persona_name", "?")
        attack = s.get("attack_strategy", "")[:65]
        vc = s.get("voice_characteristics", {})
        accent = vc.get("accent", "none")
        noise = vc.get("background_noise", "quiet")
        print(f"    [{d}] {name}")
        print(f"        attack: {attack}")
        print(f"        voice: accent={accent}, noise={noise}")

    # --- Score progression ---
    display.section("SCORE PROGRESSION")

    baseline = experiments[0] if experiments else None
    keeps = [e for e in experiments if e.get("status") == "keep"]
    discards = [e for e in experiments if e.get("status") == "discard"]
    skips = [e for e in experiments if e.get("status") == "skip"]

    if baseline:
        print(f"    Baseline:   {baseline['score']:.3f}  "
              f"(CSAT={baseline.get('csat', 0):.0f}, "
              f"pass={baseline.get('pass_rate', 0):.0%})")

    best_exp = max(experiments, key=lambda e: e.get("score", 0)) if experiments else None
    if best_exp:
        print(f"    Best:       {best_exp['score']:.3f}  "
              f"(CSAT={best_exp.get('csat', 0):.0f}, "
              f"pass={best_exp.get('pass_rate', 0):.0%}, "
              f"exp {best_exp.get('experiment', '?')})")

    if baseline and best_exp:
        delta = best_exp["score"] - baseline["score"]
        pct = delta / max(baseline["score"], 0.001) * 100
        print(f"    Delta:      {'+' if delta >= 0 else ''}{delta:.3f} "
              f"({'+' if pct >= 0 else ''}{pct:.1f}%)")

    print()
    print(f"    Kept:       {len(keeps)}")
    print(f"    Discarded:  {len(discards)}")
    print(f"    Skipped:    {len(skips)}")

    # --- Experiment log ---
    display.section("EXPERIMENTS")

    for exp in experiments:
        n = exp.get("experiment", "?")
        score = exp.get("score", 0)
        status = exp.get("status", "?")
        desc = exp.get("description", "")[:70]
        ctype = exp.get("change_type", "")
        prompt_len = exp.get("prompt_len", 0)

        icon = "+" if status == "keep" else "-" if status == "discard" else "~"
        tag = f"[{ctype}]" if ctype and ctype != "baseline" else ""

        print(f"    {icon} exp {n:>2d}  {score:.3f}  {status:7s}  "
              f"{prompt_len:>5d} chars  {tag} {desc}")

    # --- Changes that stuck ---
    kept_changes = [e for e in experiments
                    if e.get("status") == "keep"
                    and e.get("experiment", 0) > 0]

    if kept_changes:
        display.section("CHANGES THAT STUCK")
        for exp in kept_changes:
            n = exp.get("experiment", "?")
            score = exp.get("score", 0)
            delta = exp.get("delta", 0)
            desc = exp.get("description", "")
            reasoning = exp.get("reasoning", "")
            arrow = "+" if delta > 0 else "="
            print(f"    exp {n}: {arrow}{abs(delta):.3f} → {score:.3f}")
            print(f"      {desc[:90]}")
            if reasoning:
                print(f"      why: {reasoning[:90]}")
            print()

    # --- Prompt diff ---
    display.section("PROMPT")
    print(f"    Original: {len(original)} chars")
    print(f"    Best:     {len(best)} chars")
    diff = len(best) - len(original)
    print(f"    Delta:    {'+' if diff >= 0 else ''}{diff} chars")

    if best and os.path.exists(best_path):
        print(f"\n    Best prompt saved at: {best_path}")
        print(f"\n    --- BEST PROMPT (first 500 chars) ---")
        print()
        for line in best[:500].split("\n"):
            print(f"    {line}")
        if len(best) > 500:
            print(f"\n    ... ({len(best) - 500} more chars, see {best_path})")

    # --- Failures discovered ---
    all_failures: set[str] = set()
    for exp in experiments:
        for r in exp.get("results", []):
            all_failures.update(r.get("failure_modes", []))

    if all_failures:
        display.section("FAILURE MODES DISCOVERED")
        for fm in sorted(all_failures):
            print(f"    - {fm}")

    # --- Generate graphs ---
    from . import graphs
    display.section("GRAPHS")
    try:
        paths = graphs.generate_research(experiments, out_dir)
        for p in paths:
            display.info(f"  {p}")
    except Exception as e:
        display.info(f"  Graph generation failed: {e}")

    display.blank()
    display.info(f"Full data: {log_path}")
    display.info(f"TSV log:   {tsv_path}")
    display.blank()
