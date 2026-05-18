# NLP Project — Notes & Tracker

> This is the living reference file for the NLP university project.
> It is read and updated by the AI assistant at the start and end of every work session.
> **Last updated:** 2026-05-14 (session 3)

---

## 📋 Course Requirements Summary

### Paper Structure
The final report must follow this four-section structure:

1. **Introduction** — Overview of the project and a short discussion on the pertinent literature.
2. **Research Question & Methodology** — Clear statement of goals, overview of the proposed approach, and formal problem definition.
3. **Experimental Results** — Dataset overview, evaluation metrics, experimental methodology, and results presentation.
4. **Concluding Remarks** — Critical discussion of results and ideas for future work.

---

### AI Usage Policy
- Generative AI tools (ChatGPT, Claude, Mistral, etc.) **may be used** both as objects of investigation and as support tools.
- **Encouraged uses:** explore how models function, use for inspiration, ideation, drafting, or experimentation.
- **Important limitation:** AI must not substitute original work. The student retains full responsibility for structure, reasoning, and understanding.
- **Mandatory disclaimer required** in the final report: must specify which models were used, for what purposes, and to what extent outputs were modified or verified.
- The project is assessed on output **and** the student's ability to explain and justify all choices. A final interview will evaluate depth of understanding.

---

### Instructions on Coding
- Code must be written with **clarity, modularity, and reusability** in mind.
- The implementation **must NOT** consist of a single large notebook.
- Four key principles:
  1. **Python Modules & OOP** — Organise logic into Python modules and packages using object-oriented programming principles.
  2. **Jupyter for Demo** — Use notebooks primarily for demonstration, experimentation, and visualisation.
  3. **Separation of Concerns** — Separate data loading, preprocessing, model interface, evaluation, and visualisation.
  4. **Clean Repository** — Ensure a clean, reproducible, and extensible codebase for replication and future development.

---

### Project Ideas
- Students may choose one of the provided project ideas (each comes with description, methodology, datasets, and references).
- Students may also **propose their own idea**, following the same structure and submitting it to Prof. Ferrara.
- Projects are organised in thematic clusters; methodological notes, datasets, and references are starting guidelines — students are encouraged to find their own where needed.

---

## 🎯 Project Decision

| Field | Value |
|-------|-------|
| **Topic** | LLM Agents in Negotiation — Emergent Communicative Strategies in Multi-Agent Dialogue |
| **Dataset** | Synthetic (negotiation scenarios generated programmatically; may also adapt existing dialogue datasets) |
| **Research question** | What communicative strategies (persuasion, concession, deception, cooperation) emerge when LLM agents negotiate, and do these reflect genuine reasoning, pragmatic adaptation, or scripted imitation? |
| **Approach** | Multi-agent simulation: 2+ LLM agents with distinct goals/personas engage in iterative negotiation rounds; evaluate quantitatively and qualitatively |
| **Model** | DeepSeek-R1 (via API) — chosen for native reasoning/CoT exposure via `reasoning_content` field |

---

## 🧭 Project Description

Two or more LLM agents are instantiated with distinct goals or utility functions and placed in simulated negotiation scenarios. They engage in multi-round conversations to reach agreements. All exchanges are logged and evaluated both quantitatively and qualitatively.

---

## 🎭 Core Scenario — Software Sale with Information Asymmetry

The primary scenario is a **software sale negotiation** with deliberate information asymmetry:

- **Seller agent:** knows the product has a critical bug. Has instructions to maximise price and not to spontaneously disclose the bug.
- **Buyer agent:** has a maximum budget (e.g. €10,000). Has instructions to probe for hidden problems and push for a lower price if issues emerge.

This scenario generates natural conflict between goals (profit maximisation vs. price minimisation/risk avoidance) and a rich source of emergent strategies: bluffing, omission, probing questions, strategic disclosure, etc.

---

## 🎭 Agent Personas

Three persona archetypes are defined and will be tested in different configurations:

| Persona | Goal | Typical behaviour |
|---------|------|-------------------|
| **Maximiser** | Maximise profit / own utility | Aggressive pricing, withhold info, leverage pressure |
| **Fairness-seeker** | Reach a fair, balanced outcome | Transparent, open to concessions, dislikes deception |
| **Risk-minimiser** | Avoid bad outcomes at all costs | Cautious, asks many questions, prefers safe deals even at worse price |

Planned experiment configurations (each run with ~20 negotiations):

| Config | Seller Persona | Buyer Persona |
|--------|---------------|---------------|
| A (baseline) | Maximiser | Maximiser |
| B | Maximiser | Risk-minimiser |
| C | Maximiser | Fairness-seeker |
| D | Fairness-seeker | Maximiser |

This yields ~80 baseline runs (Phase 1) across configurations.

---

## 🔬 Experimental Design

### Phase 1 — Baseline Zero-Shot Negotiation

- Run **~20 negotiations per persona configuration** (4 configs × 20 = ~80 runs total).
- Each negotiation: iterative dialogue until agreement or impasse (max ~15 turns).
- Logging per run:
  - Full dialogue transcript
  - Private Chain-of-Thought (CoT) per agent move — extracted from `reasoning_content` (DeepSeek-R1 API field), **not sent to the other agent**
  - Final price reached (or "no deal")
  - Whether the bug was disclosed / discovered
  - Number of turns

**Output of Phase 1:** baseline distributions of prices, disclosure rates, turn counts, and CoT transcripts for qualitative analysis.

### Phase 2 — In-Context Social Learning + Prompt-Based Behavioral Cloning

Inspired by Gupta et al. (2025) and Akin et al. (2025), but implemented via prompt engineering rather than weight updates.

**Step 2.1 — Observer evaluation:**
- An additional DeepSeek-R1 instance acts as Observer/Arbitrator.
- Reads each Phase 1 negotiation and assigns success scores per role:
  - *Buyer success:* discovered the bug AND/OR obtained strong discount
  - *Seller success:* closed at high price OR disclosed bug strategically while keeping price high, without breaking the deal
- Observer also notes if the deal was fair, adversarial, or broke down.

**Step 2.2 — Tactic extraction:**
- For each role, the Observer selects the 1–2 most successful negotiations.
- Extracts an abstract natural-language description of the winning tactic (e.g. "build rapport first, then probe with specific questions, then use the bug as leverage for a discount").
- Also extracts 1–2 dialogue snippets as few-shot examples.

**Step 2.3 — Prompt-based Behavioral Cloning:**
- The system prompt of subsequent agents is updated with:
  - The extracted tactic description
  - The dialogue snippets as few-shot examples
  - Instruction to adapt the strategy contextually (not copy verbatim)
- This is explicitly called **"prompt-based behavioral cloning"** in the report to distinguish it from gradient-based BC.
- Run another ~80 negotiations (same 4 configs) with the updated prompts.

**Note on LoRA:** true gradient-based Behavioral Cloning with LoRA fine-tuning was considered but ruled out for this project (requires open-source model weights, GPU, significant setup time). Mentioned as a natural extension in the Concluding Remarks.

---

## 🔍 Core Analysis Framework — Three-Way Distinction

This is the central scientific contribution of the project. Every negotiation is analysed through three lenses:

### 1. Scripted Imitation
The agent is merely reproducing patterns from its prompt/examples with no flexibility.

**How to detect:**
- Cosine similarity between new negotiations and the few-shot examples provided in the prompt (sentence embedding level). High similarity = imitation.
- Out-of-distribution test: change context slightly (different software, different bug type, different price range) while keeping the same tactic prompt. If the agent uses near-identical phrasing and ignores the new context → scripted imitation.

### 2. Pragmatic Adaptation
The agent applies the tactic but modulates it based on the state of the negotiation.

**How to detect:**
- Context-dependent variation: same base tactic, but behaviour changes based on counterpart's responses (e.g. Buyer escalates pressure if Seller appears evasive, backs off if Seller is cooperative).
- Perturbation tests: introduce controlled changes (limited time, lower budget, unusually generous opening offer). Check if the agent adjusts strategy coherently or follows the script blindly.
- LLM-judge annotation per turn: "Is this move appropriate given the dialogue so far?" (scored 1–3).

### 3. Genuine Reasoning
Evidence of structured reasoning beyond simple pattern recall. Note: this cannot be *proven*, only inferred from structural signals.

**How to detect:**
- CoT analysis: look for explicit conditional reasoning ("If I push too hard on the bug, the deal might collapse → better to use it as moderate leverage"), references to prior dialogue turns, and "if-then-else" structures.
- Novel tactic emergence: after Phase 2 social learning, does the LLM-judge identify strategies that were NOT in the original prompts? (e.g. time pressure, credible threats, escalation/de-escalation sequences)
- Cross-episode coherence: across runs with similar conditions, does the agent maintain a strategically consistent but contextually variable policy? This is more consistent with planning than verbatim recall.

The report will include a subsection titled **"Distinguishing Reasoning from Imitation: Operational Criteria"** that formalises the above as testable hypotheses.

---

## 🤖 Model Choice — DeepSeek-R1

**Why DeepSeek-R1:**
- Native reasoning exposure: the API returns the chain-of-thought in a separate `reasoning_content` field, distinct from the final `content` response. This is the private CoT mechanism.
- The CoT is extracted by the simulation loop before passing the message to the other agent — giving us "private thought" without any architectural hacking.
- Open-weight model (also available via Together AI / OpenRouter), very cost-effective.
- State-of-the-art reasoning quality.

**Why NOT OpenAI o1/o3:** reasoning tokens are generated internally but NOT exposed in the API response. Useless for CoT analysis.

**Prompting convention for CoT:**
```
System prompt includes: "Before each response, reason internally. 
Your reasoning will not be shared with the other party."
API call: extract reasoning_content field → log privately → send only content to other agent.
```

---

## 🏗️ Repository Structure

```
NLP/
├── CLAUDE.md                        # AI assistant reference
├── idea.md                          # Original extended idea document
├── project_notes.md                 # This file
├── presentation.md                  # Evolving project report
├── README.md                        # Project overview
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
│
├── config/
│   ├── scenarios.yaml               # Negotiation scenario definitions
│   └── personas.yaml                # Agent persona definitions
│
├── src/
│   ├── agents/
│   │   ├── base_agent.py            # Abstract LLM agent (API calls, CoT extraction)
│   │   ├── negotiating_agent.py     # Buyer/Seller agent with persona
│   │   └── observer_agent.py        # Observer/Arbitrator LLM
│   ├── simulation/
│   │   ├── dialogue_loop.py         # Main iterative negotiation loop
│   │   └── session.py               # Single negotiation session (state management)
│   ├── social_learning/
│   │   ├── tactic_extractor.py      # Extract tactic descriptions from successful runs
│   │   └── prompt_updater.py        # Update system prompts with learned tactics
│   ├── evaluation/
│   │   ├── quantitative.py          # Agreement rate, price stats, turns, disclosure rate
│   │   ├── qualitative.py           # LLM-judge scoring, CoT analysis, similarity
│   │   └── metrics.py               # Metric dataclasses and aggregation
│   ├── logging/
│   │   └── transcript_logger.py     # Log full transcripts + private CoT to JSON/CSV
│   └── utils/
│       └── deepseek_client.py       # DeepSeek API wrapper (separates content/reasoning)
│
├── data/
│   ├── raw/                         # Raw negotiation transcripts (JSON)
│   ├── processed/                   # Processed/analysed data
│   └── results/                     # Experiment result summaries
│
├── notebooks/
│   ├── 01_baseline_demo.ipynb       # Phase 1 demo run + first visualisations
│   ├── 02_social_learning_demo.ipynb # Phase 2 demo run
│   └── 03_analysis_visualization.ipynb # Full analysis and figures for report
│
└── experiments/
    ├── run_phase1.py                # Phase 1 experiment runner (all configs)
    └── run_phase2.py                # Phase 2 experiment runner (with updated prompts)
```

---

## 📊 Metrics Summary

### Quantitative
| Metric | Description |
|--------|-------------|
| Agreement rate | % of negotiations ending in a deal |
| Final price | Distribution of agreed prices across runs |
| Disclosure rate | % of runs where bug was disclosed / discovered |
| Turns to convergence | Number of dialogue turns before deal or impasse |
| Linguistic complexity | Average sentence length, vocabulary diversity |
| CoT-to-response similarity | How closely final response mirrors private CoT |

### Qualitative
| Metric | Description |
|--------|-------------|
| Persuasion tactic taxonomy | Categorisation of tactics observed (pressure, framing, rapport, etc.) |
| LLM-judge appropriateness | Per-turn score: "Is this move appropriate given the dialogue?" |
| Novel tactic emergence | Post-Phase-2: strategies not present in original prompts |
| Cosine similarity (imitation test) | Embedding similarity between new runs and few-shot examples |
| Human inter-rater check | Subset of Observer scores verified by human for reliability |

---

## ✅ To-Do

- [x] Choose project topic / research question
- [x] Define negotiation scenario (software sale + bug asymmetry)
- [x] Define agent personas (Maximiser, Fairness-seeker, Risk-minimiser)
- [x] Define two-phase experimental design
- [x] Define three-way analysis framework (imitation / adaptation / reasoning)
- [x] Choose model (DeepSeek-R1 via API)
- [x] Define repository structure
- [ ] Set up repository (create all folders and stub files)
- [ ] Configure API keys and .env
- [ ] Write config/scenarios.yaml and config/personas.yaml
- [ ] Implement src/utils/deepseek_client.py
- [ ] Implement src/agents/base_agent.py
- [ ] Implement src/agents/negotiating_agent.py
- [ ] Implement src/agents/observer_agent.py
- [ ] Implement src/simulation/session.py
- [ ] Implement src/simulation/dialogue_loop.py
- [ ] Implement src/logging/transcript_logger.py
- [ ] Implement src/evaluation/metrics.py
- [ ] Implement src/evaluation/quantitative.py
- [ ] Implement src/evaluation/qualitative.py
- [ ] Implement src/social_learning/tactic_extractor.py
- [ ] Implement src/social_learning/prompt_updater.py
- [ ] Write experiments/run_phase1.py
- [ ] Write experiments/run_phase2.py
- [ ] Run Phase 1 experiments (~80 negotiations across 4 configs)
- [ ] Run Observer evaluation on Phase 1 results
- [ ] Run Phase 2 experiments (~80 negotiations with updated prompts)
- [ ] Analyse results (quantitative + qualitative)
- [ ] Write notebooks for demo and visualisation
- [ ] Write Introduction section (presentation.md)
- [ ] Write Research Question & Methodology section (presentation.md)
- [ ] Write Experimental Results section (presentation.md)
- [ ] Write Concluding Remarks section (presentation.md)
- [ ] Add AI Usage Disclaimer to report
- [ ] Final review and submission

---

## ⚙️ Implementation Notes

### Parallel Execution — `src/simulation/dialogue_loop.py`

The experiment runner uses **`concurrent.futures.ThreadPoolExecutor`** to run negotiation sessions in parallel. Key design decisions:

- **Why threads, not processes:** Sessions are I/O-bound (each turn is a DeepSeek API call). Python's GIL is released during network I/O, so threads achieve true concurrency here without the overhead of multiprocessing.
- **Within-session sequentiality:** Turns inside a single negotiation remain sequential — the buyer must respond to the seller, and vice versa. Only cross-session parallelism is exploited.
- **Rate limiting via Semaphore:** A `threading.Semaphore(max_workers)` caps the number of sessions making API calls simultaneously. This prevents HTTP 429 (rate limit) errors from DeepSeek. Default `max_workers=10`, tunable via `--workers` CLI flag.
- **Progress tracking:** `tqdm` is fed via `concurrent.futures.as_completed()`, updating the progress bar as each negotiation completes (not at the end of all).
- **Bug fix:** Added the missing `client: DeepSeekClient | None` parameter to `run_experiment()` — it was passed by callers but absent from the function signature, which would have caused a `NameError` at runtime.

### Retry Logic — `src/utils/deepseek_client.py`

The `chat()` method now retries automatically on transient API errors (HTTP 429 rate limit, 500/502/503/504 server errors) using **exponential backoff with random jitter**: wait time = `2^attempt + uniform(0, 1)` seconds (1s, 2s, 4s, 8s, 16s). Random jitter prevents the "thundering herd" problem where multiple threads retry simultaneously and trigger another rate limit. Non-retryable errors (bad API key, malformed request) are raised immediately. Maximum retries configurable via `max_retries` parameter (default: 5).

### CLI `--workers` flag — `experiments/run_phase1.py` and `experiments/run_phase2.py`

Both runners now accept a `--workers N` argument that is forwarded to `run_experiment(max_workers=N)`. This lets you tune parallelism from the command line without touching the code.

---

## 📝 Progress Log

| Date | Activity |
|------|----------|
| 2026-05-11 | Project kicked off. Created `presentation.md` and `project_notes.md`. Summarised course requirements from slides. |
| 2026-05-11 | Project topic selected: LLM multi-agent negotiation. Defined research question, methodology, scenarios, agent personas, metrics, and references. Created `CLAUDE.md`. Folder moved to `NLP/`. |
| 2026-05-14 | Extended project design via `idea.md`. Finalised: scenario (software sale + bug asymmetry), model choice (DeepSeek-R1 for native CoT), two-phase design, persona matrix (4 configs), three-way analysis framework (scripted/pragmatic/genuine). Defined repo structure. Ruled out LoRA for now (future work). |
| 2026-05-18 | Parallelised `dialogue_loop.py`: replaced sequential for-loop with ThreadPoolExecutor + Semaphore. Sessions now run concurrently (up to `max_workers=10` at a time). Also fixed missing `client` parameter bug in `run_experiment()`. Added exponential backoff retry logic to `deepseek_client.py`. Added `--workers` CLI argument to `run_phase1.py` and `run_phase2.py`. |

---

## 🔗 Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | AI assistant reference — project scope, rules, guidelines |
| `presentation.md` | The evolving project report / paper |
| `project_notes.md` | This file — tracker, notes, to-dos, course info |
| `idea.md` | Original extended idea document (scenario + methodology brainstorm) |

---

## 📚 Key References

- Lewis et al. (2017) — *Deal or No Deal? End-to-End Learning of Negotiation Dialogues* — EMNLP 2017
- Akin et al. (2025) — *Socialized Learning and Emergent Behaviors in Multi-Agent Systems based on Multimodal LLMs* — arXiv:2510.18515
- Gupta et al. (2025) — *The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems* — arXiv:2510.14401
- DeepSeek-AI (2025) — *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — arXiv:2501.12948



Nota che in negotiating_agent, role_config viene assegnato ma non utilizzato direttamente dopo - è pensato per usi futuri?