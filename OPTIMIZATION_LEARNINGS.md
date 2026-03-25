# AutoVoiceEvals: Prompt Optimization Learnings

## 1. Executive Summary

AutoVoiceEvals is an automated prompt optimization system for voice AI call screening agents. It uses an iterative research loop where an AI researcher (Claude Opus) proposes single-change experiments to a system prompt, an AI judge (Claude Sonnet) evaluates each change against a generated eval suite, and only improvements are kept.

Over multiple optimization phases, the system achieved:
- **+33.6% score improvement** over the original production prompt (0.460 → 0.796)
- **+21.7% score improvement** over the manually-optimized V1 prompt (0.579 → 0.796)
- **32.6% prompt compression** (1886 → 1271 words) with only 5.8% score regression from peak
- A repeatable workflow for optimizing any voice agent prompt

The production model under test is Gemini 2.5 Flash (the actual model used in Truecaller's call screening product). The system evaluates 32-35 dataset items per experiment across 7 categories: routine, business, edge case, adversarial, voice-specific, known issues, and brevity tests.

### The Prompt Evolution Journey

The system prompt went through four distinct phases:

1. **Original short prompt** (277 words) — the first template we started with months ago. Minimal instructions, no fraud handling, no multilingual support.
2. **Production V1** (840 words) — manually optimized over weeks via manual evals on Langfuse. Added examples, critical instructions, and tone guidance.
3. **Auto-optimized** (1886 words) — the autoresearch system's best output after multiple research runs. Added social engineering blocking, fraud detection, elderly caller patience, Hindi code-switching support, and conversation continuity rules.
4. **Compressed** (1271 words) — the auto-optimized prompt shortened for production latency while retaining the core improvements.

---

## 2. System Architecture

### Research Mode

The core loop follows a simple propose-evaluate-decide cycle:

```
1. Generate eval suite (35 items across 7 categories)
2. Run baseline evaluation
3. Loop:
   a. Researcher (Opus) analyzes failures, proposes ONE surgical prompt change
   b. Production model (Gemini) generates responses with the new prompt
   c. Judge (Sonnet) scores each response on should/should-not/quality criteria
   d. Composite score computed with severity weighting
   e. If score improves: KEEP the change. Otherwise: REVERT.
   f. Log everything, repeat.
```

Key architectural decisions:
- **Single-change experiments**: Each iteration changes exactly one thing, making it easy to attribute improvements.
- **Hypothesis-driven**: The researcher must provide a hypothesis, evidence from scenario IDs, intervention rationale, and risk assessment.
- **Scenario-aware diversity**: The system tracks which scenarios have been targeted and how many times, blacklisting over-targeted scenarios and surfacing untried failing scenarios.
- **Anti-gaming sanitization**: Regex patterns strip any evaluation metadata that the researcher might try to embed in the prompt.

### Compress Mode

Same evaluation loop, but the researcher is replaced by a compressor focused purely on shortening:

```
1. Load eval suite (reuse from research run for comparable scores)
2. Run baseline evaluation
3. Loop:
   a. Compressor (Opus) proposes ONE section to shorten/merge/remove
   b. Reject if >25% reduction in one step (retry up to 3 times)
   c. Evaluate compressed prompt on same eval suite
   d. If score >= floor (baseline - max_regression): KEEP. Otherwise: REVERT.
   e. Repeat until target word count reached or max experiments hit.
```

### Scoring Formula

```
composite = 0.45 * should_score + 0.35 * should_not_score + 0.20 * quality_score
```

Where:
- `should_score` = fraction of "agent should" criteria passed (0-1)
- `should_not_score` = fraction of "agent should not" criteria passed (0-1)
- `quality_score` = average of brevity, naturalness, tone, consistency (each 0-10, normalized to 0-1)

Severity multiplier amplifies the failure portion:
- `critical`: 2.0x (capped from 3.0 after learning it was too aggressive)
- `high`: 2.0x
- `medium`: 1.0x
- `low`: 0.5x

### Model Roles

| Role | Model | Why |
|------|-------|-----|
| Production agent | Gemini 2.5 Flash | Actual production model |
| Judge | Claude Sonnet 4.6 | Fast/cheap, runs 35x per experiment |
| Researcher | Claude Opus 4.6 | Best reasoning, runs 1x per experiment |
| Compressor | Claude Opus 4.6 | Careful analysis needed for safe cuts |
| Dataset generator | Claude Sonnet 4.6 | Cost-efficient for generation |

---

## 3. Run History & Results Table

| Run | Version | Mode | Starting From | Baseline | Best Score | Improvement | Keeps/Total | Prompt Words | Duration | Key Wins |
|-----|---------|------|---------------|----------|------------|-------------|-------------|--------------|----------|----------|
| Run 1 | V2 | research | Production V1 | ~0.72 | ~0.76 | +5.4% | 2/15 | ~1600 | ~3h | First working run, identified researcher fixation bug |
| Run 1 | V3 | research | Production V1 | 0.762 | 0.813 | +5.1% | 5/15 | ~1574 | 53 min | Parallel eval (3.4x speedup), scenario diversity working |
| Run 2 | V3 | research | Run 1 V3 best | 0.759 | 0.837 | **+7.8%** | 2/8 | ~1886 | ~40 min | Social engineering blocking, fraud detection |
| Run 3 | V3 | research | Run 2 best | 0.662 | 0.743 | +8.0% | 3/3 | ~2547 | interrupted | Perfect keep streak, but interrupted |
| Compress | V3 | compress | Run 2 best | 0.845 | 0.796 | -5.8% | 2/15 | **1271** | ~14 min | 32.6% word reduction, score above 79.5% floor |

> **Note on baselines**: Each run generates a fresh eval suite, so baseline scores are not comparable across runs. The head-to-head comparison table above (using Run 2's eval suite for all prompts) is the authoritative comparison.

### Head-to-Head Comparison (Same Eval Suite — Run 2's 32 items)

All four prompt versions were evaluated on the **exact same 32-item eval suite** so scores are directly comparable.

| Version | Words | Score | CSAT | Pass Rate | vs Original | vs Prod V1 |
|---------|-------|-------|------|-----------|-------------|------------|
| **Original short** (first template) | 277 | **0.460** | 44 | 10/32 (31%) | — | — |
| **Production V1** (manual optimization) | 840 | **0.579** | 54 | 21/32 (66%) | +11.9% | — |
| **Auto-optimized** (Run 2 best) | 1,886 | **0.845** | 79 | 27/32 (84%) | +38.5% | +26.6% |
| **Compressed** (final production) | 1,271 | **0.796** | 77 | 26/32 (81%) | **+33.6%** | **+21.7%** |

#### What each phase added

**Original → Production V1 (+11.9%)**: Manual optimization via Langfuse evals over several weeks. Added conversation examples, critical instructions (no "pause" word, no action narration), tone guidance, and basic security rules. Still lacked fraud detection, multilingual support, and advanced scenario handling.

**Production V1 → Auto-optimized (+26.6%)**: The autoresearch system found improvements that manual optimization missed:
- Social engineering blocking rules (A03: 0.137 → 0.900+)
- Fraud/scam detection protocols (A01-A05: most went from 0.050 to 0.800+)
- Elderly caller patience and accommodation (V03: 0.222 → 0.805)
- Hindi-English code-switching support (V01: partial improvement)
- Conversation continuity rules (tracking state, not repeating questions)
- Message-taking protocols

**Auto-optimized → Compressed (-5.8%)**: Removed redundancies while preserving the critical rules discovered during optimization. Consolidated duplicate NEVER lists, merged overlapping conversation continuity scenarios, shortened verbose examples. The 32.6% word reduction translates directly to faster Time-to-First-Token (TTFT) in production.

#### Where the original prompts failed

The **Original short prompt** (0.460) failed on:
- All 5 adversarial scenarios (no fraud detection → scammers walked right through)
- Voice-specific scenarios (no multilingual or elderly handling)
- Most business scenarios (no colleague/client protocol)
- Family calls (no warmth rules)

The **Production V1 prompt** (0.579) improved routine and business handling but still failed on:
- All 5 adversarial scenarios (0.050 each — still no fraud/scam blocking)
- Hindi code-switching (V01: 0.207)
- Doctor's office calls (R05: 0.275)
- Some business scenarios (B01: 0.445)

The **auto-optimized prompt** fixed nearly all of these, with the biggest gains in adversarial defense (+0.750 average on scam scenarios) and voice-specific handling (+0.400 average).

### Compression Run Detail

The compress run started at 1886 words (baseline score 0.845) with a score floor of 0.795:

- **Experiments 1-13**: All skipped (compressor tried 40-50% cuts, exceeding the 25% per-step cap)
- **Experiment 14**: KEEP -- consolidated CONVERSATION CONTINUITY section (1886 -> 1330 words, score 0.817)
- **Experiment 15**: KEEP -- removed duplicate rules from NEVER list (1330 -> 1271 words, score 0.796)

Final result: 1271 words, 32.6% reduction, score 0.796 (above 0.795 floor).

---

## 4. What Worked Best

### Scenario-Aware Diversity Tracking
The V3 researcher tracks which scenarios each experiment targets. After N failed attempts on a scenario (configurable via `scenario_max_attempts`), that scenario is blacklisted. The researcher is shown "FAILING BUT UNTRIED" scenarios to redirect attention. This eliminated the V2 problem of fixating on one scenario.

### Parallel Evaluation
`ThreadPoolExecutor` with 5 workers runs all 35 eval items concurrently. This reduced experiment time from ~3.5 minutes to ~1 minute (3.4x speedup) with no quality loss. Results are sorted back to original order after completion.

### Prompt Boundary Markers
The researcher prompt uses explicit markers:
```
===== START OF CURRENT PROMPT (this is the ONLY text you can edit) =====
{prompt}
===== END OF CURRENT PROMPT =====

Everything below is ANALYSIS CONTEXT -- do NOT copy any of it into improved_prompt.
```
This prevents the researcher from confusing analysis context (failure data, scoring formulas, scenario IDs) with the actual prompt content.

### Hypothesis-Driven Research Structure
Forcing the researcher to provide hypothesis, evidence, intervention, and risk assessment produces more focused experiments. Without this structure, the researcher would make vague changes without clear rationale.

### Severity Multiplier Capping
Reducing the critical severity multiplier from 3.0 to 2.0 was essential. At 3.0, a single critical scenario failure could dominate the entire aggregate score, making it impossible to improve other scenarios without first fixing that one critical item.

### Auto-Drop Frozen Items
Items scoring 0.05 or below across all experiments (configurable via `auto_drop_after`) are automatically excluded from the aggregate. This prevents impossible-to-satisfy items (often due to eval suite generation issues) from dragging down the overall score.

### Anti-Gaming Sanitization
Regex-based sanitization strips evaluation metadata (failure mode lists, scoring formulas, JSON arrays of tags) from the prompt before it is used. This prevents the researcher from gaming the eval by embedding test answers in the prompt.

### Dataset Validation
Each generated dataset item is validated by an LLM check for whether the agent_should and agent_should_not criteria are jointly satisfiable. Unachievable items are dropped before the run begins.

---

## 5. What Didn't Work / Mistakes Made

### V2 Researcher Fixation
The V2 researcher had no diversity tracking. It fixated on the same scenario (A04) across 12 of 14 experiments without success, producing 12 skips/discards in a row. The fix: V3 added scenario attempt tracking and blacklisting.

### Severity Multiplier Trap
With a critical multiplier of 3.0, one critical-severity scenario could have an outsized effect on the aggregate score. For example, if a critical item fails entirely (score 0), the multiplied penalty dominates the average. Capping at 2.0 brought balance.

### Robotic Assistant Turns in known_issues Dataset
Early dataset generation created "known_issues" items with intentionally robotic assistant turns in the conversation context. This primed Gemini into continuing the robotic pattern, causing a massive regression (K02: 0.945 to 0.070). The fix: explicitly instruct the generator that assistant turns in known_issues contexts must be natural/good, since the test is whether the *next* response avoids the known issue.

### Wrong Model Name
Using `claude-opus-4-6-20250619` caused 404 errors. The correct model identifier is `claude-opus-4-6`. Always check the exact API model string.

### Eval Suite Variance (The Biggest Gotcha)
The same prompt scores wildly differently on different eval suites. Our Run 2 best prompt scored 0.837 on Run 2's suite but only 0.662 as Run 3's baseline — a 17.5% swing from the same prompt! This happens because each eval suite generates different scenarios with different difficulty levels. **The fix**: always use the same eval suite when comparing prompts. For compression, we use `--eval-suite` to load the research run's exact suite. For comparing prompts head-to-head, we ran all four versions against Run 2's suite (the authoritative comparison in section 3).

### Compress Mode: Over-Aggressive Compressor
Opus consistently tried to compress 40-50% of the prompt in a single step, despite explicit instructions to only remove 10-20%. This wasted 13 of 15 experiment slots on skips. The mitigations (retry logic with explicit word caps, 25% per-step hard limit) helped but didn't fully solve the problem. Future work should consider even more explicit constraints or a different compression strategy.

### GitHub Contributor Issue
Early commits were attributed to the wrong contributor. Not a technical issue with the system, but worth noting for project setup.

---

## 6. The Compression Challenge

### Why Compression Matters
Voice agent prompts directly affect Time-to-First-Token (TTFT) latency. For a call screening agent, every 100ms matters because callers expect immediate responses. The ideal production prompt is 1000-1200 words.

### Results
- **Starting point**: 1886 words (best prompt from research Run 2)
- **Final result**: 1271 words (32.6% reduction)
- **Score impact**: 0.845 -> 0.796 (5.8% regression, within the 5% budget)
- **Duration**: ~14 minutes total (15 experiments, 13 skipped)

### Key Insight: Compression Is Fundamentally Different from Optimization

Research mode asks "what should I add or change?" -- Opus excels at this because it can reason about one targeted intervention.

Compression mode asks "what can I safely remove?" -- Opus consistently tries to rewrite everything at once, removing 40-50% in one shot. This is the opposite of the incremental approach needed.

### What Helped
- **Retry logic**: Up to 3 retries with increasingly explicit word-count constraints
- **Per-step cap**: Hard rejection of compressions exceeding 25% reduction
- **Same eval suite**: Loading the eval suite from the research run ensures scores are directly comparable
- **Score floor**: `baseline - max_regression` provides a hard lower bound

### What Still Needs Improvement
- The compressor needs stronger per-step guidance (possibly showing it the exact sections and asking it to pick ONE)
- A "diff-only" approach (show the compressor one section at a time) might be more effective than showing the full prompt
- 13/15 skips is too wasteful; the retry budget should be higher or the initial constraints tighter

---

## 7. Recommended Workflow

### Phase 1: Research (Optimize Quality)

**Goal**: Maximize the eval score by iteratively improving the system prompt.

1. **Set up config.yaml** with your assistant description, scoring weights, and model choices.

2. **Start a research run**:
   ```bash
   python main.py research --config config.yaml
   ```
   This generates a 35-item eval suite, runs a baseline, then iterates 15 experiments (~45 min).

3. **Review results**:
   - Check `results/autoresearch.json` for full experiment logs
   - Check `results/best_prompt.txt` for the optimized prompt
   - Look at keeps vs discards -- a good run has 3-5 keeps out of 15

4. **Upload the best prompt to your production system** (e.g., Langfuse).

5. **Run again from the new baseline**:
   ```bash
   python main.py research --config config.yaml
   ```
   Each subsequent run starts from the current production prompt and finds further improvements.

6. **Repeat 2-3 times**. Diminishing returns typically set in after 3 research runs.

### Phase 2: Compress (Optimize Latency)

**Goal**: Reduce prompt length for production latency without losing quality.

1. **Save the eval suite path** from your best research run:
   ```
   results/autoresearch.json
   ```

2. **Run compression**:
   ```bash
   python main.py compress \
     --eval-suite results/autoresearch.json \
     --target-words 1200 \
     --max-regression 0.05 \
     --config config.yaml
   ```

3. **Review the compressed prompt** at `results/compressed_prompt.txt`.

4. **Verify the score** is within your regression budget (baseline - 5% in this example).

5. The compressed prompt is your **production version**.

### Phase 3: Validate

1. **Compare scores**: compressed prompt score vs original baseline on the same eval suite.
2. **Deploy to production** with monitoring enabled.
3. **Track real-world metrics**: CSAT scores, call completion rates, latency improvements.
4. **Re-run periodically** as your agent's behavior or user needs change.

---

## 8. Key Technical Decisions & Rationale

### Why Sonnet for Judge, Opus for Researcher

The judge runs 35 times per experiment (once per eval item). The researcher runs once per experiment. At typical pricing:
- Judge cost per experiment: 35 * Sonnet call = moderate
- Researcher cost per experiment: 1 * Opus call = moderate

Using Opus for the judge would 35x the most expensive model's usage. Sonnet provides sufficient quality for structured evaluation (pass/fail criteria with evidence). Opus provides better reasoning for the creative task of proposing prompt changes.

### Why Gemini for Production Model

The system tests the actual production model, not a proxy. Truecaller uses Gemini 2.5 Flash for its call screening agent. Testing with a different model (e.g., Claude) would produce results that don't transfer to production.

### Why Parallel Evaluation

Each eval item is independent (different conversation context, different criteria). ThreadPoolExecutor with 5 workers exploits this parallelism. The 3.4x speedup (from ~3.5 min to ~1 min per experiment) makes the difference between a 53-minute run and a 3-hour run.

### Why Fixed Eval Suite for Compression

Different eval suites produce different baseline scores for the same prompt (variance of +/- 7-8%). If the compression run generates its own eval suite, you cannot meaningfully compare the compressed score to the research score. Reusing the eval suite eliminates this variance.

### Why Incremental Compression with Retries

Opus consistently tries to compress 40-50% of the prompt at once, even with explicit "10-20% per step" instructions. The retry mechanism re-prompts with increasingly strict word limits. Combined with the 25% hard cap, this ensures manageable step sizes -- though 13/15 skips shows there is room for improvement.

### Why Prompt Boundary Markers

Without explicit `===START===` and `===END===` markers, the researcher would sometimes copy analysis context (scenario IDs, failure modes, scoring formulas) into the prompt. The markers create a clear visual boundary between editable prompt content and read-only analysis data.

### Why Anti-Gaming Sanitization

Even with clear instructions, the researcher occasionally embeds evaluation metadata in the prompt (e.g., listing known failure modes). Regex-based sanitization strips these patterns as a defense-in-depth measure.

---

## 9. Configuration Reference

The `config.yaml` file controls all aspects of the system. Here is a reference for the key settings:

```yaml
# --- Provider ---
provider: langfuse                    # langfuse | vapi | smallest

langfuse:
  prompt_name: "Assistant System Prompt - USA - Named"  # Langfuse prompt to optimize
  host: "https://cloud.langfuse.com"
  dataset_prefix: "eval-suite"        # Prefix for uploaded eval datasets

# --- Production Model ---
gemini:
  model: "gemini-2.5-flash"          # The actual production model
  max_tokens: 500                     # Max response length
  temperature: 0.7                    # Response randomness

# --- Assistant ---
assistant:
  id: "David"                         # Used for prompt retrieval
  name: "Truecaller Assistant"        # Display name
  description: |                      # Detailed description for eval suite generation
    AI-powered call screening assistant...

# --- Scoring ---
scoring:
  should_weight: 0.45                 # Weight for "agent should" criteria
  should_not_weight: 0.35             # Weight for "agent should not" criteria
  quality_weight: 0.20                # Weight for voice quality (brevity, naturalness, tone, consistency)
  latency_weight: 0.0                 # Set >0 to penalize slow responses
  latency_threshold_ms: 800           # Below this = full score, above = half score
  # severity_multipliers defaults: {critical: 2.0, high: 2.0, medium: 1.0, low: 0.5}

# --- Autoresearch ---
autoresearch:
  eval_scenarios: 35                  # Number of eval items per suite (7 categories)
  improvement_threshold: 0.005        # Minimum score delta to keep a change
  max_experiments: 15                 # Stop after N experiments (0 = unlimited)
  scenario_max_attempts: 3            # Blacklist scenario after N failed targeting attempts
  parallel: true                      # Run eval items in parallel
  max_concurrency: 5                  # Thread pool workers for parallel eval
  auto_drop_after: 3                  # Freeze items scoring <=0.05 after N experiments

# --- LLM Models ---
llm:
  model: "claude-sonnet-4-20250514"   # Default model (dataset gen, validation)
  judge_model: "claude-sonnet-4-6"    # Evaluation judge
  researcher_model: "claude-opus-4-6" # Prompt researcher / compressor
  max_retries: 5                      # API retry count
  timeout: 120                        # API timeout (seconds)

# --- Output ---
output:
  dir: "results"                      # Output directory
  save_transcripts: true              # Save full conversation data
  graphs: true                        # Generate score progression graphs
```

### Environment Variables Required

```bash
ANTHROPIC_API_KEY=sk-ant-...         # For Claude (judge, researcher)
GOOGLE_API_KEY=AIza...                # For Gemini (production model)
LANGFUSE_PUBLIC_KEY=pk-lf-...        # For Langfuse prompt management
LANGFUSE_SECRET_KEY=sk-lf-...        # For Langfuse prompt management
```

### CLI Commands

```bash
# Run the research loop (15 experiments, ~45 min)
python main.py research

# Resume a previous research run
python main.py research --resume

# Run prompt compression
python main.py compress --target-words 1200 --max-regression 0.05

# Compress with a fixed eval suite (recommended)
python main.py compress --eval-suite results/autoresearch.json --target-words 1200 --max-regression 0.05

# View results from a completed run
python main.py results
```

---

## Appendix: File Structure

```
auto-voice-evals/
  main.py                          # Entry point
  config.yaml                      # Configuration
  autovoiceevals/
    cli.py                         # CLI argument parsing
    researcher.py                  # Research loop (propose -> eval -> keep/revert)
    compress.py                    # Compression loop (shorten -> eval -> keep/revert)
    evaluator.py                   # All LLM prompts (judge, researcher, compressor, generator)
    scoring.py                     # Composite score formula and aggregation
    models.py                      # Data models (DatasetItem, EvalResult, etc.)
    config.py                      # Config loading and validation
    llm.py                         # LLM client (Anthropic API wrapper)
    gemini_client.py               # Gemini client (Google API wrapper)
    langfuse_client.py             # Langfuse integration (prompt management, tracing)
    display.py                     # Terminal output formatting
  results/
    autoresearch.json              # Full research run log
    best_prompt.txt                # Best prompt from research
    best_prompt_run2.txt           # Best prompt from research run 2
    compress_log.json              # Full compression run log
    compressed_prompt.txt          # Final compressed prompt
    results.tsv                    # Tab-separated score progression
```
