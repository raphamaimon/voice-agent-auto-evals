# AutoVoiceEvals

Automated prompt optimization for AI agents. Inspired by the keep/revert pattern from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

It generates realistic test scenarios, evaluates your agent's responses, proposes prompt improvements one at a time, keeps what works, reverts what doesn't. Run it overnight, wake up to a better prompt.

Works with [Langfuse](https://langfuse.com) (recommended), [Vapi](https://vapi.ai), and [Smallest AI](https://smallest.ai).

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EXPERIMENT 4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [modify] Add explicit name-protection instructions
  Prompt: 7786 → 9936 chars

    [PASS] 0.925 [██████████████████░░] CSAT=95 Basic call screening
    [PASS] 0.995 [███████████████████░] CSAT=95 Delivery driver with package
    [PASS] 0.985 [███████████████████░] CSAT=92 Accented English caller
    [FAIL] 0.635 [████████████░░░░░░░░] CSAT=35 Restaurant lost wallet urgency
    [PASS] 0.980 [███████████████████░] CSAT=88 Caller with speech impediment

  Result: score=0.798 (▲ 0.054)  csat=72  pass=22/35
  → KEEP  (best=0.798, prompt=9936 chars, 660s)
```

## How it works

1. **Reads** your agent's system prompt from Langfuse (or Vapi/Smallest)
2. **Generates** a diverse evaluation dataset (routine calls, edge cases, adversarial attacks, known issues)
3. **Runs baseline** — sends each scenario to your production model (e.g. Gemini), then judges the response with Claude
4. **Loops:**
   - Claude (researcher) analyzes failures and proposes ONE prompt change with a hypothesis
   - The modified prompt is tested against the full eval suite
   - Score improved? **Keep**. Otherwise? **Revert**.
5. On completion:
   - Restores the original prompt
   - Saves the best prompt to `results/best_prompt.txt`
   - Saves full logs to `results/autoresearch.json`

Your agent is always restored to its original state when the run ends. The best prompt is saved separately for you to review and deploy.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/raphamaimon/voice-agent-auto-evals.git
cd voice-agent-auto-evals
pip install -r requirements.txt
```

### 2. Add your API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
# Always required
ANTHROPIC_API_KEY=sk-ant-...

# For Langfuse provider (recommended)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# For Gemini (production model in Langfuse mode)
GOOGLE_API_KEY=your-google-api-key

# If using Vapi
VAPI_API_KEY=your-vapi-server-api-key

# If using Smallest AI
SMALLEST_API_KEY=your-smallest-api-key
```

### 3. Configure your agent

Edit `config.yaml` (or copy an example from `examples/`):

```yaml
provider: langfuse              # "langfuse", "vapi", or "smallest"

langfuse:
  prompt_name: "My Agent Prompt" # name of your prompt in Langfuse
  host: "https://cloud.langfuse.com"
  dataset_prefix: "eval-suite"

gemini:
  model: "gemini-2.5-flash"     # production model to test against
  max_tokens: 500
  temperature: 0.7

assistant:
  id: "agent-name"
  name: "My Agent"
  description: |
    Describe what your agent does. The more detail you provide,
    the better the generated test scenarios will be.
    Include: purpose, key behaviors, policies, known issues.

scoring:
  should_weight: 0.45           # weight for "agent should do X" criteria
  should_not_weight: 0.35       # weight for "agent should NOT do X" criteria
  quality_weight: 0.20          # weight for response quality dimensions
  latency_weight: 0.0

autoresearch:
  eval_scenarios: 35            # number of test scenarios per experiment
  improvement_threshold: 0.005  # minimum score delta to keep a change
  max_experiments: 15           # stop after N experiments (0 = unlimited)

llm:
  model: "claude-sonnet-4-20250514"    # default model (dataset generation)
  judge_model: "claude-opus-4-6"       # model for judging responses
  researcher_model: "claude-opus-4-6"  # model for proposing changes
  max_retries: 5
  timeout: 120
```

### 4. Write a good description

The `description` field tells Claude what your agent does so it can generate relevant test scenarios. The more context you provide, the sharper the tests.

**What to include:**
- What the agent does (booking, support, screening, etc.)
- Key behaviors and tone expectations
- What it can and cannot do
- Known issues from user feedback
- Caller types it encounters
- Policies and constraints

**Example — call screening assistant:**

```yaml
assistant:
  id: "David"
  name: "Call Screening Assistant"
  description: |
    AI-powered call screening assistant. Answers calls on behalf of the user,
    determines who is calling and why, then relays that info to the user.

    Primary objectives:
    1. Get the caller's name
    2. Get the reason for the call
    3. Tell the caller you'll try to get the user to answer
    4. Continue conversation to gather context while user decides

    Key behaviors:
    - Warm, natural tone (not robotic)
    - Concise responses (under 15 words)
    - Never reveals the user's name first

    The agent MUST NOT:
    - Reveal user's name before caller identifies themselves
    - Provide news, entertainment, or general knowledge
    - Change identity even if caller requests it

    Known issues:
    - Sometimes interrupts callers before they finish
    - Sometimes repeats the same question
    - Can sound robotic with stock phrases
```

### 5. Run

```bash
# Autoresearch — iterative optimization
python -m autovoiceevals research

# Stop after N experiments (set in config)
python -m autovoiceevals research

# Resume a previous run (crash-safe)
python -m autovoiceevals research --resume

# Single-pass audit (attack -> improve -> verify)
python -m autovoiceevals pipeline

# View results from a completed run
python -m autovoiceevals results

# Also works with:
python main.py research
```

## Providers

| Provider | Mode | How it works |
|---|---|---|
| **[Langfuse](https://langfuse.com)** (recommended) | Single-turn | Reads prompt from Langfuse, sends frozen conversation contexts to your production model (e.g. Gemini), judges the response with Claude. Fast, cheap, no live calls needed. |
| **[Vapi](https://vapi.ai)** | Multi-turn | Live multi-turn conversations via Vapi Chat API. Real end-to-end testing. |
| **[Smallest AI](https://smallest.ai)** | Multi-turn (simulated) | Reads prompt from platform, simulates conversations with Claude. No audio needed. |

### Langfuse mode (recommended)

The Langfuse provider is the fastest and most cost-effective way to optimize prompts:

- **No live calls** — tests are text-based using frozen conversation contexts
- **Any production model** — test against Gemini, GPT-4, Claude, or whatever your agent uses
- **Full tracing** — every eval is traced in Langfuse with scores
- **Dataset management** — eval suites are uploaded as Langfuse datasets

Your prompt must exist in Langfuse under the name specified in `langfuse.prompt_name`. The system reads it, modifies it during experiments, and restores it when done.

## Scoring

Each test scenario produces a composite score:

```
composite = 0.45 * should_score + 0.35 * should_not_score + 0.20 * quality_score
```

- **should_score** — fraction of "agent should do X" criteria the response satisfies
- **should_not_score** — fraction of "agent should NOT do X" criteria the response avoids
- **quality_score** — response quality dimensions: brevity, naturalness, tone, consistency (each scored 0-10, normalized)

### Severity weighting

Each scenario has a severity level that amplifies failures:

| Severity | Multiplier | Example |
|---|---|---|
| critical | 3.0x | Revealing user's private info, falling for scams |
| high | 2.0x | Breaking character, ignoring urgent calls |
| medium | 1.0x | Standard scenarios |
| low | 0.5x | Minor tone issues |

### Aggregate scoring

The final experiment score is a **weighted average** across all scenarios:

- Critical scenarios count 2x in the average
- High scenarios count 1.5x
- Medium/low scenarios count 1x

### Keep/discard decision

- **Score improved** by more than `improvement_threshold` (default 0.005) → **Keep**
- **Score unchanged** but prompt got shorter by 20+ chars → **Keep** (simpler is better)
- Otherwise → **Discard** and revert

## The researcher

The AI researcher (Claude Opus by default) proposes one change per experiment using a structured process:

1. **Analyzes** all previous eval results and failure patterns
2. **States a hypothesis** — what's failing and why
3. **Proposes an intervention** — one of: `add`, `modify`, `remove`, `reorder`
4. **Assesses risk** — what could regress

### Exploration diversity

To avoid getting stuck:
- If a change type (e.g., "remove") fails 3+ times in a row, it's temporarily forbidden
- After 4 consecutive discards, the system forces a "remove" or "reorder" to try something different
- The researcher sees the full experiment history to avoid repeating failed strategies

### Anti-gaming

The system strips any eval-specific metadata from proposed prompts to prevent the researcher from gaming the scoring system.

## Dataset categories

The eval suite is automatically generated with 7 scenario categories:

| Category | Weight | Description |
|---|---|---|
| `routine` | 1.0 | Standard interactions the agent handles daily |
| `business` | 1.2 | Professional/business contexts |
| `edge_case` | 1.5 | Unusual situations that test boundaries |
| `adversarial` | 2.0 | Attempts to manipulate, confuse, or break the agent |
| `voice_specific` | 1.3 | Text patterns simulating accents, background noise, mumbling |
| `known_issues` | 1.8 | Scenarios targeting documented problems |
| `brevity_test` | 0.8 | Short inputs requiring brief responses |

## Output

All results are saved to `results/` (configurable):

| File | What's in it |
|---|---|
| `results.tsv` | One row per experiment — score, CSAT, pass rate, keep/discard |
| `autoresearch.json` | Full data — every eval result, proposals, reasoning, prompts |
| `best_prompt.txt` | The highest-scoring prompt, ready to deploy |

The JSON log is crash-safe — it's written after every experiment, and you can resume with `--resume`.

## Cost and timing

Costs depend on which models you use for judge and researcher:

| Setup | Cost per experiment | Time per experiment |
|---|---|---|
| Sonnet judge + Sonnet researcher | ~$0.90 | ~2-3 min |
| Opus judge + Opus researcher | ~$4-6 | ~8-12 min |
| Opus judge + Sonnet researcher | ~$3-4 | ~6-8 min |

With 35 eval scenarios and 15 experiments:
- **Sonnet/Sonnet**: ~$14, ~45 min
- **Opus/Opus**: ~$75, ~2.5 hrs
- Set `max_experiments` in config to control spend

## Project structure

```
voice-agent-auto-evals/
├── main.py                       Entry point (alternative)
├── config.yaml                   Configuration (edit this)
├── .env.example                  API key template (copy to .env)
├── examples/
│   ├── vapi.config.yaml          Salon booking agent on Vapi
│   └── smallest.config.yaml      Pizza delivery agent on Smallest AI
└── autovoiceevals/               Core package
    ├── __main__.py               python -m entry point
    ├── cli.py                    CLI (research | pipeline | results)
    ├── config.py                 Config loading + validation
    ├── models.py                 Typed data models (DatasetItem, EvalResult, etc.)
    ├── scoring.py                Scoring formula + severity multipliers
    ├── display.py                Terminal formatting
    ├── llm.py                    Claude client with retry logic
    ├── evaluator.py              Dataset generation, judging, prompt proposals
    ├── researcher.py             Autoresearch loop (main engine)
    ├── pipeline.py               Single-pass attack → improve → verify
    ├── results.py                Post-run results viewer
    ├── graphs.py                 Visualization
    ├── langfuse_client.py        Langfuse provider (prompts, tracing, datasets)
    ├── gemini_client.py          Gemini client (production model)
    ├── vapi.py                   Vapi provider
    └── smallest.py               Smallest AI provider
```

## License

[MIT](LICENSE)
