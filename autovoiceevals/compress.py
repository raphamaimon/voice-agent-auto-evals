"""Prompt compression loop.

Takes the best-performing prompt and iteratively shortens it
while maintaining evaluation score above a configurable floor.

Goal: reduce ~1800 words to ~1000-1200 words for production
voice agent latency optimization.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from .config import Config
from .models import DatasetItem, EvalResult, ExperimentRecord, Metrics
from .scoring import composite_score, aggregate
from .evaluator import Evaluator
from .llm import LLMClient
from . import display


def _build_provider(cfg: Config, llm_client: LLMClient | None = None):
    """Create the voice platform client based on config.provider."""
    if cfg.provider == "langfuse":
        from .langfuse_client import LangfuseClient
        return LangfuseClient(
            langfuse_public_key=cfg.langfuse_public_key,
            langfuse_secret_key=cfg.langfuse_secret_key,
            langfuse_host=cfg.langfuse.host,
            llm_client=llm_client,
            prompt_name=cfg.langfuse.prompt_name,
        )
    raise ValueError("Compress mode only supports langfuse provider")


def _build_gemini(cfg: Config):
    from .gemini_client import GeminiClient
    return GeminiClient(
        api_key=cfg.google_api_key,
        model=cfg.gemini.model,
        max_tokens=cfg.gemini.max_tokens,
        temperature=cfg.gemini.temperature,
    )


def _eval_single_turn(provider, gemini_client, evaluator, cfg, system_prompt, item):
    """Run one dataset item: Gemini response -> judge -> score."""
    response = provider.run_single_turn(system_prompt, item, gemini_client)

    try:
        ev = evaluator.evaluate_single_turn(
            item.conversation_context, response, item,
        )
    except Exception:
        ev = {
            "csat_score": 50, "passed": False, "summary": "Eval failed",
            "agent_should_results": [], "agent_should_not_results": [],
            "issues": [], "failure_modes": ["EVAL_ERROR"],
            "strengths": [], "weaknesses": [],
        }

    sr = ev.get("agent_should_results", [])
    snr = ev.get("agent_should_not_results", [])
    voice_quality = ev.get("voice_quality", {})
    severity = ev.get("severity_override") or item.severity

    score, s_score, sn_score, q_score = composite_score(
        sr, snr, cfg.scoring,
        voice_quality=voice_quality,
        severity=severity,
    )

    result = EvalResult(
        scenario_id=item.id,
        persona=item.description or item.category,
        score=score,
        csat_score=ev.get("csat_score", 50),
        passed=ev.get("passed", False),
        should_score=s_score,
        should_not_score=sn_score,
        failure_modes=ev.get("failure_modes", []),
        issues=ev.get("issues", []),
        summary=ev.get("summary", ""),
        strengths=ev.get("strengths", []),
        weaknesses=ev.get("weaknesses", []),
        agent_response=response,
        voice_quality=voice_quality,
        quality_score=q_score,
        weight=item.weight,
    )

    if hasattr(provider, 'score_trace'):
        provider.score_trace(result)

    return result


def _run_eval_parallel(provider, gemini_client, evaluator, cfg, prompt, suite, max_workers=5):
    """Run eval suite in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_item = {
            pool.submit(
                _eval_single_turn,
                provider, gemini_client, evaluator, cfg, prompt, item,
            ): item
            for item in suite
        }
        for future in as_completed(future_to_item):
            try:
                result = future.result()
            except Exception as e:
                item = future_to_item[future]
                result = EvalResult(
                    scenario_id=item.id,
                    persona=item.description or item.category,
                    score=0.0, csat_score=50, passed=False,
                    failure_modes=["EVAL_ERROR"],
                    summary=f"Parallel eval error: {str(e)[:100]}",
                    weight=item.weight,
                )
            display.eval_result_line(result)
            results.append(result)

    id_order = {item.id: i for i, item in enumerate(suite)}
    results.sort(key=lambda r: id_order.get(r.scenario_id, 999))
    return results


def _json_default(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _save_log(log: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "compress_log.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=_json_default)


def run(cfg: Config, target_words: int = 1200, max_regression: float = 0.03,
        min_score: float = 0.0, eval_suite_path: str = "") -> None:
    """Run the prompt compression loop.

    Args:
        cfg: Configuration object.
        target_words: Target word count for compressed prompt.
        max_regression: Maximum allowed score drop from baseline (e.g., 0.03 = 3%).
        min_score: Absolute minimum score floor (e.g., 0.80). Takes priority
                   over max_regression if it results in a higher floor.
    """
    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    assistant_id = cfg.assistant.id

    max_experiments = cfg.autoresearch.max_experiments or 15

    # Build clients
    llm = LLMClient(
        cfg.anthropic_api_key,
        model=cfg.llm.model,
        timeout=cfg.llm.timeout,
        max_retries=cfg.llm.max_retries,
    )
    evaluator = Evaluator(
        llm,
        judge_model=cfg.llm.judge_model or None,
        researcher_model=cfg.llm.researcher_model or None,
    )
    provider = _build_provider(cfg, llm_client=llm)
    gemini_client = _build_gemini(cfg)

    max_workers = cfg.autoresearch.max_concurrency

    def run_eval(prompt, suite):
        if cfg.autoresearch.parallel:
            return _run_eval_parallel(
                provider, gemini_client, evaluator, cfg,
                prompt, suite, max_workers=max_workers,
            )
        results = []
        for item in suite:
            r = _eval_single_turn(
                provider, gemini_client, evaluator, cfg, prompt, item,
            )
            display.eval_result_line(r)
            results.append(r)
        return results

    # --- Header ---
    display.header("AutoVoiceEvals — Prompt Compression Mode")
    display.info("Shorten → Eval → Keep if score holds → Repeat")

    # Get current prompt from Langfuse
    original_prompt = provider.get_system_prompt(assistant_id)
    best_prompt = original_prompt
    original_words = len(original_prompt.split())

    display.blank()
    display.info(f"Assistant: {cfg.assistant.name or assistant_id}")
    display.info(f"Prompt: {len(original_prompt)} chars ({original_words} words)")
    display.info(f"Target: ~{target_words} words")
    display.info(f"Max score regression: {max_regression:.1%}")
    display.info(f"Judge model: {cfg.llm.judge_model or cfg.llm.model}")
    display.info(f"Compressor model: {cfg.llm.researcher_model or cfg.llm.model}")
    display.info(f"Production model: {cfg.gemini.model}")
    display.info(f"Max experiments: {max_experiments}")
    if cfg.autoresearch.parallel:
        display.info(f"Parallel eval: {max_workers} workers")

    # --- Load or generate eval suite ---
    n_eval = cfg.autoresearch.eval_scenarios
    display.blank()

    if eval_suite_path:
        # Reuse eval suite from a previous run for comparable scores
        display.info(f"Loading eval suite from: {eval_suite_path}")
        with open(eval_suite_path) as f:
            prev_run = json.load(f)
        eval_suite = [DatasetItem.from_dict(s) for s in prev_run["eval_suite"]]
        display.info(f"Loaded {len(eval_suite)} items from previous run")
        display.dataset_item_list(eval_suite)
    else:
        display.info("Generating eval suite...")
        eval_suite = evaluator.generate_dataset(
            n_eval, cfg.assistant.description, original_prompt,
        )
        display.info(f"{len(eval_suite)} dataset items generated:")
        display.dataset_item_list(eval_suite)

        # Validate
        display.info("Validating dataset items...")
        valid_items = []
        for item in eval_suite:
            try:
                check = evaluator.validate_dataset_item(item)
                if check.get("achievable", True):
                    valid_items.append(item)
                else:
                    display.info(f"  Dropped {item.id}: {check.get('reason', 'unachievable')[:60]}")
            except Exception:
                valid_items.append(item)
        if valid_items:
            eval_suite = valid_items
            display.info(f"{len(eval_suite)} items passed validation")

    # Upload dataset
    if hasattr(provider, 'upload_dataset'):
        dataset_name = f"compress-{datetime.now().strftime('%Y%m%d-%H%M')}"
        provider.upload_dataset(dataset_name, eval_suite)
        display.info(f"Uploaded to Langfuse dataset: {dataset_name}")

    # --- Baseline ---
    display.section("BASELINE EVALUATION")
    display.blank()

    baseline_results = run_eval(original_prompt, eval_suite)
    baseline_metrics = aggregate(baseline_results)
    baseline_score = baseline_metrics.avg_score
    # Score floor: the higher of (baseline - max_regression) or the absolute min_score
    score_floor = max(baseline_score - max_regression, min_score)

    display.blank()
    display.info(
        f"Baseline: score={baseline_score:.3f}  csat={baseline_metrics.avg_csat:.0f}  "
        f"pass={baseline_metrics.n_passed}/{baseline_metrics.n_total}"
    )
    if min_score > 0 and min_score > baseline_score - max_regression:
        display.info(f"Score floor: {score_floor:.3f} (absolute minimum)")
    else:
        display.info(f"Score floor: {score_floor:.3f} (baseline {baseline_score:.3f} - {max_regression:.1%})")

    # Tracking
    history: list[ExperimentRecord] = []
    best_score = baseline_score
    last_eval = baseline_results

    history.append(ExperimentRecord(
        number=0, score=baseline_score, status="keep",
        description="baseline", prompt_len=len(best_prompt),
    ))

    full_log = {
        "meta": {
            "version": "compress-1.0",
            "mode": "compression",
            "assistant": cfg.assistant.name or assistant_id,
            "original_words": original_words,
            "target_words": target_words,
            "max_regression": max_regression,
            "score_floor": score_floor,
            "baseline_score": baseline_score,
            "started": datetime.now().isoformat(),
        },
        "eval_suite": eval_suite,
        "original_prompt": original_prompt,
        "experiments": [{
            "experiment": 0,
            "timestamp": datetime.now().isoformat(),
            "description": "baseline",
            "score": baseline_score,
            "csat": baseline_metrics.avg_csat,
            "prompt_len": len(original_prompt),
            "prompt_words": original_words,
            "status": "keep",
            "results": baseline_results,
        }],
    }

    # --- Compression loop ---
    display.blank()
    display.info(f"Starting compression loop ({max_experiments} experiments).")
    display.blank()

    results_path = os.path.join(out_dir, "compress_results.tsv")
    with open(results_path, "w") as f:
        f.write("experiment\tscore\tcsat\tpass_rate\tprompt_len\twords\tstatus\tdescription\n")
        f.write(
            f"0\t{baseline_score:.6f}\t{baseline_metrics.avg_csat:.1f}\t"
            f"{baseline_metrics.pass_rate:.3f}\t{len(original_prompt)}\t"
            f"{original_words}\tkeep\tbaseline\n"
        )

    experiment = 0

    try:
        while experiment < max_experiments:
            current_words = len(best_prompt.split())

            # Check if we've hit the target
            if current_words <= target_words:
                display.info(
                    f"Target reached! {current_words} words ≤ {target_words} target."
                )
                break

            experiment += 1
            t0 = time.time()

            display.section(f"COMPRESSION {experiment}")
            display.info(
                f"  Current: {current_words} words → Target: {target_words} words "
                f"({current_words - target_words} to go)"
            )

            # 1. AI proposes a compression (with retries for over-aggressive)
            max_step_pct = 0.25  # max 25% reduction per step
            max_retries_for_step = 3
            proposal = None
            for attempt in range(max_retries_for_step):
                extra_constraint = ""
                if attempt > 0:
                    max_words_to_remove = int(current_words * max_step_pct)
                    extra_constraint = (
                        f"\n\nCRITICAL: Your last attempt removed too many words. "
                        f"You MUST remove at most {max_words_to_remove} words "
                        f"(max {max_step_pct:.0%} of {current_words}). "
                        f"Pick ONE small section to shorten, not multiple sections."
                    )
                proposal = evaluator.propose_prompt_compression(
                    best_prompt, last_eval, history,
                    baseline_score=best_score,
                    target_words=target_words,
                    score_floor=score_floor,
                    extra_constraint=extra_constraint,
                )
                new_prompt = proposal.get("improved_prompt", best_prompt)
                new_words = len(new_prompt.split())
                reduction_pct = (current_words - new_words) / current_words if current_words > 0 else 0
                if reduction_pct <= max_step_pct:
                    break
                if attempt < max_retries_for_step - 1:
                    display.info(
                        f"  [retry {attempt+1}] {reduction_pct:.0%} reduction too aggressive, retrying..."
                    )

            description = proposal.get("description", "unknown")
            technique = proposal.get("technique", "?")
            reasoning = proposal.get("reasoning", "")
            new_prompt = proposal.get("improved_prompt", best_prompt)
            new_words = len(new_prompt.split())

            display.info(f"  [{technique}] {description}")
            display.info(f"  Reasoning: {reasoning[:100]}...")
            display.info(f"  Prompt: {current_words} → {new_words} words ({current_words - new_words:+d})")

            # Reject over-aggressive compression (>25% in one step) after retries
            reduction_pct = (current_words - new_words) / current_words
            if reduction_pct > max_step_pct:
                display.info(
                    f"  → SKIP (too aggressive: {reduction_pct:.0%} reduction, max {max_step_pct:.0%} per step)"
                )
                history.append(ExperimentRecord(
                    number=experiment, score=best_score,
                    status="skip", description=f"too aggressive: {description}",
                    prompt_len=len(new_prompt),
                    change_type=technique,
                ))
                continue

            # Skip if no actual compression
            if new_words >= current_words:
                display.info("  → SKIP (no compression achieved)")
                history.append(ExperimentRecord(
                    number=experiment, score=best_score,
                    status="skip", description=description,
                    prompt_len=len(new_prompt),
                    change_type=technique,
                ))
                continue

            # Skip if prompt grew
            if len(new_prompt.strip()) >= len(best_prompt.strip()):
                display.info("  → SKIP (prompt not shorter)")
                history.append(ExperimentRecord(
                    number=experiment, score=best_score,
                    status="skip", description=description,
                    prompt_len=len(new_prompt),
                    change_type=technique,
                ))
                continue

            # 2. Apply and evaluate
            provider.update_prompt(assistant_id, new_prompt)
            display.blank()

            eval_results = run_eval(new_prompt, eval_suite)
            m = aggregate(eval_results)
            new_score = m.avg_score
            delta = new_score - best_score

            dt = time.time() - t0

            # 3. Keep or revert
            if new_score >= score_floor:
                status = "keep"
                compression_pct = (1 - new_words / original_words) * 100
                best_prompt = new_prompt
                best_score = new_score
                last_eval = eval_results
                display.blank()
                display.info(
                    f"  Result: score={new_score:.3f} ({delta:+.3f})  "
                    f"csat={m.avg_csat:.0f}  pass={m.n_passed}/{m.n_total}"
                )
                display.info(
                    f"  → KEEP  ({new_words} words, {compression_pct:.0f}% compressed from original, {dt:.0f}s)"
                )
            else:
                status = "discard"
                provider.update_prompt(assistant_id, best_prompt)
                display.blank()
                display.info(
                    f"  Result: score={new_score:.3f} ({delta:+.3f}) — below floor {score_floor:.3f}"
                )
                display.info(
                    f"  → DISCARD  (score dropped too much, {dt:.0f}s)"
                )

            # 4. Log
            with open(results_path, "a") as f:
                f.write(
                    f"{experiment}\t{new_score:.6f}\t{m.avg_csat:.1f}\t"
                    f"{m.pass_rate:.3f}\t{len(new_prompt)}\t{new_words}\t"
                    f"{status}\t{description[:80]}\n"
                )

            history.append(ExperimentRecord(
                number=experiment, score=new_score, status=status,
                description=description, prompt_len=len(new_prompt),
                change_type=technique,
            ))

            full_log["experiments"].append({
                "experiment": experiment,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "technique": technique,
                "reasoning": reasoning,
                "prompt_len": len(new_prompt),
                "prompt_words": new_words,
                "score": new_score,
                "delta": delta,
                "csat": m.avg_csat,
                "pass_rate": m.pass_rate,
                "status": status,
                "duration_s": dt,
                "results": eval_results,
                "prompt": new_prompt if status == "keep" else None,
            })

            _save_log(full_log, out_dir)
            display.blank()

    except KeyboardInterrupt:
        display.header("STOPPED (Ctrl+C)")

    # --- Final report ---
    final_words = len(best_prompt.split())
    compression_pct = (1 - final_words / original_words) * 100
    keeps = sum(1 for h in history if h.status == "keep") - 1  # exclude baseline
    discards = sum(1 for h in history if h.status == "discard")
    skips = sum(1 for h in history if h.status == "skip")

    display.header("COMPRESSION COMPLETE")
    display.blank()
    display.info(f"Experiments: {experiment} ({keeps} keeps, {discards} discards, {skips} skips)")
    display.info(f"Original: {original_words} words ({len(original_prompt)} chars)")
    display.info(f"Compressed: {final_words} words ({len(best_prompt)} chars)")
    display.info(f"Compression: {compression_pct:.1f}% reduction")
    display.info(f"Score: {baseline_score:.3f} → {best_score:.3f} ({best_score - baseline_score:+.3f})")
    display.info(f"Score floor was: {score_floor:.3f}")

    if final_words > target_words:
        display.info(f"⚠ Did not reach target ({final_words} > {target_words} words)")
    else:
        display.info(f"✓ Target reached ({final_words} ≤ {target_words} words)")

    # Restore original prompt in Langfuse
    display.blank()
    display.info("Restoring original prompt in Langfuse...")
    provider.update_prompt(assistant_id, original_prompt)

    # Save compressed prompt
    compressed_path = os.path.join(out_dir, "compressed_prompt.txt")
    with open(compressed_path, "w") as f:
        f.write(best_prompt)
    display.info(f"Compressed prompt saved: {compressed_path}")

    # Save final log
    full_log["meta"]["ended"] = datetime.now().isoformat()
    full_log["meta"]["total_experiments"] = experiment
    full_log["meta"]["final_score"] = best_score
    full_log["meta"]["final_words"] = final_words
    full_log["meta"]["compression_pct"] = compression_pct
    full_log["best_prompt"] = best_prompt
    _save_log(full_log, out_dir)

    if hasattr(provider, 'flush'):
        provider.flush()

    display.info(f"Results: {results_path}")
    display.info(f"Full log: {os.path.join(out_dir, 'compress_log.json')}")
    display.blank()
