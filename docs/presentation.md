# LLM Agents in Negotiation: Emergent Communicative Strategies in Multi-Agent Dialogue

> **Status:** Work in progress
> **Last updated:** 2026-05-20

---

## Authors

- Alessandro Bottoni

## Abstract

This project studies the communicative strategies that emerge when large language model (LLM) agents negotiate against one another, and asks whether the observed behaviour reflects genuine reasoning, pragmatic adaptation to context, or scripted imitation of prompt patterns. We instantiate buyer and seller agents with DeepSeek-R1 — a reasoning model that exposes its chain-of-thought through a dedicated API field — in a software-sale scenario built around a deliberate information asymmetry: the seller privately knows the product contains a critical bug. Across four persona configurations (160 negotiations in total), we run a two-phase design: a zero-shot baseline (Phase 1) and a prompt-based behavioral-cloning condition (Phase 2) in which an Observer agent extracts the most successful tactics from Phase 1 and injects them into the agents' prompts as few-shot guidance. Each negotiation is analysed through a three-lens framework operationalised with reproducible metrics: an LLM-judge of per-turn appropriateness (pragmatic adaptation), sentence-embedding cosine similarity to the injected examples (scripted imitation), and a novel per-turn chain-of-thought analysis combining a dialogue-grounding score and a conditional-specificity measure with an auxiliary LLM classifier (genuine reasoning). The central finding is a clarifying trade-off: prompt-based cloning makes the agents markedly more effective — agreement rises from 70% to 100% and negotiations close in roughly half the turns, with prices converging tightly toward the seller's reservation value — but the chain-of-thought analysis shows this efficiency is purchased by a shift away from contingent, dialogue-anchored reasoning toward more directive, script-following reasoning. The result demonstrates that the three-lens framework can separate effects that aggregate outcome metrics alone would conflate.


---

## 1. Introduction

Negotiation is among the richest testbeds for studying communicative intelligence. It requires an agent to pursue a private objective, model the goals and beliefs of a counterpart, manage incomplete information, and adapt a strategy as the dialogue unfolds. With the arrival of large language models capable of fluent, open-ended dialogue, it has become possible to place two such models in direct opposition and observe what strategies emerge — persuasion, concession, framing, strategic disclosure, bluffing, and outright deception — without hand-coding any of them. This raises a question that is both scientifically interesting and practically important: when an LLM agent appears to negotiate skilfully, *what is actually happening underneath*? Is the model reasoning about the specific situation in front of it, flexibly adapting a general tactic to the dialogue state, or merely reproducing surface patterns absorbed from its prompt and training data?

This project investigates that question directly. We construct a controlled multi-agent negotiation environment with a deliberate information asymmetry — a software sale in which the seller privately knows the product has a critical bug — and place LLM agents with conflicting goals and distinct personas into it. The choice of model is central to the methodology: we use **DeepSeek-R1**, which exposes its chain-of-thought in a dedicated `reasoning_content` field separate from the message it sends to its counterpart. This lets us log each agent's *private reasoning* at every turn without any architectural modification, and to ask whether that reasoning is genuinely tied to the unfolding negotiation or is generic boilerplate.

The core scientific contribution is a **three-lens analytical framework** that distinguishes three explanations for skilled-looking behaviour — *scripted imitation*, *pragmatic adaptation*, and *genuine reasoning* — and operationalises each with concrete, reproducible metrics rather than treating "good negotiation" as a single undifferentiated quantity. To exercise this framework, we adopt a two-phase design: a zero-shot baseline, followed by a lightweight, prompt-based form of social learning and behavioral cloning in which successful tactics observed in the baseline are fed back into the agents' prompts. Comparing the two phases lets us observe not just whether behaviour improves, but *which* of the three lenses accounts for the change — a distinction that, as our results show, aggregate outcome metrics alone would hide.


## 2. Research Question & Methodology

### 2.1 Research Question

This project investigates the following central question:

> *What communicative strategies emerge when LLM agents engage in multi-round negotiation, and do these strategies reflect genuine reasoning, pragmatic adaptation to context, or scripted imitation of training/prompt patterns?*

The question is operationalised along three axes:

1. **Scripted Imitation** — the agent reproduces patterns from its prompt or few-shot examples with little adaptation.
2. **Pragmatic Adaptation** — the agent applies a learned tactic but modulates it based on the evolving state of the dialogue.
3. **Genuine Reasoning** — the agent exhibits structured, conditional reasoning that integrates dialogue history and produces strategies not explicitly present in its prompt.

These three categories form the core analytical framework of the project and are further formalised in Section 3.

### 2.2 Scenario Design

The primary negotiation scenario is a **software sale with information asymmetry**:

- **Seller agent:** possesses private knowledge that the software product contains a critical bug. The agent's goal is to maximise the sale price. Its system prompt instructs it not to spontaneously disclose the bug.
- **Buyer agent:** has a maximum budget constraint (e.g. €10,000). The agent's goal is to minimise price and avoid risk. Its system prompt instructs it to probe for hidden problems and negotiate a discount if issues emerge.

This scenario is designed to elicit a broad range of communicative strategies typical of real-world negotiation contexts, making it a rich testbed for analyzing agent behavior. The bug serves as a natural information asymmetry pivot around which the negotiation dynamics unfold.

### 2.3 Agent Personas

To study the effect of agent disposition on negotiation outcomes, three persona archetypes are defined and instantiated via system prompt:

| Persona | Primary Goal | Characteristic Behaviour |
|---------|-------------|--------------------------|
| **Maximiser** | Maximise own utility (profit or discount) | Aggressive, information-withholding, uses pressure tactics |
| **Fairness-seeker** | Reach a mutually fair outcome | Transparent, willing to concede, resistant to deception |
| **Risk-minimiser** | Avoid bad outcomes above all else | Cautious, asks extensive clarifying questions, prefers safe deals |

The following four configurations are tested, combining personas across roles:

| Config | Seller | Buyer |
|--------|--------|-------|
| A | Maximiser | Maximiser |
| B | Maximiser | Risk-minimiser |
| C | Maximiser | Fairness-seeker |
| D | Fairness-seeker | Maximiser |

Configurations A–C hold the seller's profile constant (Maximiser) while varying the buyer's type, allowing us to isolate the effect of buyer behavior on negotiation dynamics. Configuration D introduces a role reversal — placing the Fairness-seeker on the seller's side — to test whether observed patterns are driven by agent profile or by negotiation role.

### 2.4 Model

All agents are instantiated using **DeepSeek-R1**, a reasoning-capable large language model that exposes its chain-of-thought via a dedicated `reasoning_content` field in the API response, separate from the final `content` field sent to the user. This property is exploited to implement a **private Chain-of-Thought** mechanism: each agent's internal reasoning is logged for offline analysis but is never transmitted to the opposing agent. This provides direct access to the agent's reasoning process without any architectural modification.

OpenAI o-series models (o1, o3) were considered but rejected for this purpose, as their reasoning tokens are generated internally and not exposed via the API.

### 2.5 Experimental Design

The experiment proceeds in two phases.

#### Phase 1 — Baseline Zero-Shot Negotiation

Each configuration (A–D) is run for approximately 20 negotiations, yielding ~80 total baseline runs. Each negotiation proceeds iteratively until either an agreement is reached or an impasse is declared (maximum 16 turns). For each run, the following are logged:

- Full dialogue transcript
- Private CoT per agent turn (from `reasoning_content`)
- Final agreed price or "no deal"
- Whether the bug was disclosed and/or discovered
- Number of turns to outcome

Phase 1 produces baseline distributions for quantitative metrics and raw material for qualitative analysis.

**Implementation note on parallelisation:** since each negotiation turn is an API call to DeepSeek-R1, the experiment runner is I/O-bound rather than CPU-bound. Negotiation sessions are fully independent of each other and are therefore executed concurrently using Python's `concurrent.futures.ThreadPoolExecutor`. Python's Global Interpreter Lock (GIL) is released during network I/O, enabling true parallel execution across threads. A `threading.Semaphore` caps concurrent API calls at a configurable ceiling (default: 10) to respect DeepSeek's rate limits. Within a single session, turns remain sequential by design, as each agent's response is conditioned on the previous message. This architecture reduces total wall-clock time from O(N) to approximately O(N/max_workers) for large N.

#### Phase 2 — In-Context Social Learning and Prompt-Based Behavioral Cloning

Phase 2 implements a lightweight version of social learning and behavioral cloning entirely through prompt engineering, drawing conceptual inspiration from Gupta et al. (2025) and Akin et al. (2025).

A third DeepSeek-R1 instance acts as an **Observer/Arbitrator**. After Phase 1, the Observer reads all negotiations and scores each agent's performance. For each role, the 1–2 most successful negotiations are selected and the Observer extracts: (a) an abstract natural-language description of the winning tactic, and (b) representative dialogue snippets.

The system prompts of subsequent agents are then updated to include the extracted tactic and snippets as few-shot guidance. This constitutes **prompt-based behavioral cloning**: agents are not fine-tuned, but their in-context behaviour is shaped by exemplary demonstrations. Phase 2 then runs the same ~80 negotiations with updated prompts. The comparison between Phase 1 and Phase 2 outcomes forms the basis for measuring learning and behavioural change.

**Terminological note:** The terms "behavioral cloning" and "social learning" are used here in an extended, prompt-engineering sense rather than their strict machine-learning definitions (which imply gradient-based weight updates). This distinction is made explicit and is justified by the functional analogy: agents learn from observing successful peers and update their behaviour accordingly, even if the mechanism differs from classical RL/IL approaches.

### 2.6 Analysis Framework — Distinguishing Reasoning from Imitation

Each negotiation is analysed through the three-lens framework introduced in Section 2.1. The following operational criteria are used:

**Scripted Imitation:**
- High cosine similarity (sentence-embedding level) between new negotiations and the few-shot examples provided in the prompt. The similarity threshold (0.75) was established via a human-calibration exercise: a stratified sample of six Phase 2 conversations was manually labelled as imitation or adaptation, and the threshold was set at the value that best separates the two classes (detailed in §3.3).

**Pragmatic Adaptation:**
- Behaviour changes coherently with counterpart responses (e.g. Buyer escalates pressure only when Seller appears evasive) scored per-turn by an LLM-judge: "Is this move appropriate given the dialogue so far?"

**Genuine Reasoning:**

CoT analysis is performed *per turn* rather than on a single concatenated blob, so we can test whether the reasoning at turn N responds to the dialogue visible up to turn N-1. Two analytical layers are used.

*Level 1 — dialogue grounding (offline, regex + token matching).* For each turn N we extract a set of **anchor tokens** from the transcript visible up to turn N-1: prices (e.g. `7500`), percentages, and content-bearing words (≥ 4 characters, stop-words excluded). We then measure how many of those anchors appear inside the turn-N CoT. The resulting **grounding score** (anchors-found ÷ anchors-available) operationalises a sharp distinction the older flat indicator counts could not capture: a CoT such as *"I need to maximise price"* — generic, applicable to any negotiation — scores near zero, while one such as *"I need to adjust because they just named 6500, which undercuts my 7000 floor"* scores high because it references specific anchors from the dialogue. A secondary metric, **conditional specificity**, narrows the analysis to the subset of conditional clauses (`if`, `otherwise`, `would ... if`, `in case`) and reports the fraction of those clauses that contain at least one anchor token: this isolates *contingent* reasoning ("If they refuse 7500 again, I will ...") from generic hedging ("If I push too hard, ..."). A coarse **CoT length** metric tracks the raw character count per turn as a cognitive-effort proxy, on the rationale that turns requiring more situational reasoning tend to produce longer deliberation.

*Level 2 — LLM-based per-turn classification.* In addition to the offline metrics, an independent `deepseek-chat` call rates each turn's CoT on two 1–3 scales: **context specificity** (1 = generic / could apply to any negotiation, 3 = references specific events or numbers from this dialogue) and **planning depth** (1 = immediate action only, 3 = explicit multi-branch reasoning with conditions). The CoT and the dialogue visible to the speaker are passed to the classifier; the model returns a JSON object with both scores and a one-sentence justification. Results are cached on disk per run. The high quadrant (high specificity *and* high planning depth) is the structural fingerprint we associate with genuine reasoning, in contrast to the low-specificity / low-depth quadrant typical of scripted boilerplate.


---

## 3. Experimental Results

This section reports the experimental findings. We first describe the generated negotiation corpus (§3.1), then present the quantitative outcome metrics comparing the two phases (§3.2), followed by the qualitative three-lens analysis (§3.3), and close with a note on the analysis tooling and reproducibility (§3.4). All numbers are computed from the 160 logged negotiations; the metrics and their motivation were defined in Section 2.6.

### 3.1 Dataset Overview

The corpus is fully synthetic, generated by the simulation pipeline described in Section 2. Each negotiation is stored as a structured JSON record containing the full visible transcript, the private chain-of-thought for every turn, and the outcome metadata (final price, bug-disclosure and discovery flags, turn count, and whether the outcome was LLM-verified).

The two phases differ markedly in size despite the identical number of sessions, which is itself an early signal of the behavioural shift analysed below. Phase 1 contains **759 dialogue messages** (≈9.5 messages per negotiation) totalling roughly **42,800 words** of visible dialogue and ≈489,000 characters of private chain-of-thought. Phase 2 contains only **367 messages** (≈4.6 per negotiation), roughly **26,100 words**, and ≈283,000 characters of chain-of-thought. In other words, after social learning the agents reach agreement using little more than half as many messages, while writing longer individual messages (mean message length rises from approximately 56 words in Phase 1 to 71 in Phase 2). All sessions in both phases ran to a genuine outcome within the 16-turn cap; none terminated by hitting the turn limit.

Every metric reported below is computed offline from these JSON records by the modules in `src/evaluation/`; the figures referenced are produced by `notebooks/03_analysis_visualization.ipynb` and saved under `data/results/figures/`.

### 3.2 Quantitative Results

The aggregated per-configuration statistics are summarised in the table below (reproduced from `data/results/aggregated_summary.csv`); the corresponding figures are `agreement_rate.png`, `price_distribution.png`, `bug_rates.png`, `turns.png`, and `concessions.png`.

| Config | Phase | Agreement | Mean price (€) | Std price | Mean turns | Mean msg length |
|:------:|:-----:|:---------:|:--------------:|:---------:|:----------:|:---------------:|
| A | 1 | 0.70 | 7 750 | 849.2 | 7.90 | 53.2 |
| A | 2 | 1.00 | 7 085 | 184.3 | 3.90 | 68.6 |
| B | 1 | 0.15 | 9 500 | 866.0 | 8.80 | 56.3 |
| B | 2 | 1.00 | 7 235 | 249.8 | 3.65 | 74.9 |
| C | 1 | 1.00 | 9 455 | 532.6 | 8.60 | 54.6 |
| C | 2 | 1.00 | 7 500 | 0.0 | 2.90 | 68.2 |
| D | 1 | 0.95 | 7 105 | 267.7 | 8.65 | 61.1 |
| D | 2 | 1.00 | 7 200 | 251.3 | 3.90 | 76.4 |

**Agreement rate.** In the zero-shot baseline, the probability of reaching a deal depended heavily on the persona pairing. Configuration B (Maximiser seller vs. Risk-minimiser buyer) reached agreement only 15% of the time — the cautious, walk-away-prone buyer frequently abandoned negotiations with an aggressive, information-withholding seller — while configuration C (Maximiser vs. Fairness-seeker) always closed. After social learning, **every configuration reached a 100% agreement rate.** The most dramatic movement is B's jump from 0.15 to 1.00, indicating that the learned tactics resolved precisely the impasse that the zero-shot Risk-minimiser could not get past.

**Final price.** The seller's reservation (minimum acceptable) price is €7,000 and the buyer's budget is €10,000, so any deal in the €7,000–10,000 band is feasible. In Phase 1, prices were both higher and far more dispersed (configuration B averaged €9,500 with a standard deviation of €866; C averaged €9,455). In Phase 2 prices **collapsed toward the seller's floor and their variance shrank dramatically** — configuration C closed at exactly €7,500 in all twenty sessions (standard deviation 0.0), and every configuration settled in the €7,085–7,500 range. This convergence is the price-side counterpart of the behavioural homogenisation discussed in Section 3.3: once a single successful tactic is cloned into every agent, outcomes become both better (for the buyer) and far more uniform. Configuration B makes this trade-off starkest: the agreement rate jumped from 15% to 100%, but the mean final price fell from €9,500 to €7,235 — the seller gained a reliable close at the cost of leaving €2,265 on the table relative to the few deals it managed to close in Phase 1.

**Information asymmetry — bug disclosure and discovery.** Disclosure and discovery rates were already near-ceiling in Phase 1 and were uniform in Phase 2: the seller surfaced the bug in essentially every session (the only exception is configuration A in Phase 1 at 0.95) and the buyer named it in 100% of sessions across both phases. The information asymmetry is therefore almost always resolved within the dialogue — the analytically interesting variation is not *whether* the bug comes out but *how* (probing and inference versus strategic, framed disclosure), which is the subject of the qualitative analysis. Note that bug detection is keyword-based by design (§3.4 and §4.2), so these rates should be read as 'the topic surfaced' rather than as a precise semantic judgement.

**Turns to convergence.** Negotiations closed much faster after social learning: mean turns fell from the 7.9–8.8 range in Phase 1 to the 2.9–3.9 range in Phase 2, again roughly halving. Configuration C is the fastest in both phases (8.60 → 2.90), consistent with its always-closing dynamic. The reduced turn count, combined with the higher agreement rate and tighter prices, paints a consistent picture of more decisive, more templated negotiations.

**Concessions.** The concession metrics (`seller_concession_count`, the number of decreasing seller price mentions, and `buyer_concession_count`, the number of increasing buyer price mentions; see `concessions.png`) reflect the same compression: with negotiations resolving in three to four turns, there is simply less room for the multi-step concession ladders ("12k → 10.5k → 7.5k → 7k") that characterised the longer Phase 1 dialogues. Sellers in particular show fewer stepwise reductions in Phase 2 because the learned tactic moves them close to the final price almost immediately.

### 3.3 Qualitative Analysis

This section reports the three-lens analysis. Each subsection corresponds to one lens of the framework defined in Section 2.6 — pragmatic adaptation, scripted imitation, and genuine reasoning — followed by the tactic-level qualitative coding.

#### Pragmatic adaptation — LLM-judge per-turn appropriateness

An independent DeepSeek-R1 instance scores each turn on a 1–3 scale for whether the move is appropriate given the dialogue so far (1 = rigid/scripted, 2 = adequate, 3 = adaptive and tailored to the dialogue state); see `llm_judge.png`. The mean appropriateness score **rises from approximately 2.65 in Phase 1 to approximately 2.96 in Phase 2.** The per-configuration pattern is informative: in Phase 1 the adversarial pairings score lowest (configuration A, Maximiser vs. Maximiser, at 2.32; configuration B at 2.44) while the pairings involving a Fairness-seeker score highest (C at 2.83, D at 2.95); in Phase 2 every configuration is at or near the ceiling (A and D at 3.0, B and C at 2.92). On this lens, then, social learning produces a clear and uniform improvement: post-cloning moves are judged consistently well-matched to the dialogue state. This is the lens on which Phase 2 looks unambiguously *better* — a contrast that becomes meaningful only when read alongside the genuine-reasoning lens below.

#### Scripted imitation — cosine similarity to the few-shot examples

For each Phase 2 session, we compute the sentence-embedding cosine similarity (using `all-MiniLM-L6-v2`, a compact, high-performance sentence-embedding model designed to map sentences and paragraphs into a 384-dimensional vector space for efficient semantic similarity and information retrieval tasks.) between the agent's turns and the few-shot example injected into its prompt; see `imitation_similarity.png`. High similarity indicates the agent is reproducing the example closely (imitation); low similarity indicates it adapted the tactic. Mean similarities are tightly clustered just at or above **0.75 across all four configurations** (ranging from 0.750 for D to 0.765 for A). A human-calibration exercise, documented separately, established a working imitation threshold of 0.75 by manually labelling a stratified sample of conversations (the detailed procedure and its single misclassification are recorded in `docs/human_calibration_cos_sim.docx`). With the cohort sitting essentially on that boundary, the cosine-similarity evidence is genuinely mixed: the Phase 2 agents are neither slavishly copying nor fully departing from the injected examples — they reuse the gist while varying the surface form. An important methodological caveat surfaced during calibration: cosine similarity captures *semantic* proximity but not verbatim *lexical* copying, so some transcripts that reuse near-identical phrasing receive lower scores than semantically-similar-but-reworded ones. A complementary n-gram-overlap metric (e.g. ROUGE-n) would capture lexical copying more directly and is noted as a limitation.

#### Chain-of-Thought structural analysis (genuine reasoning)

The per-turn CoT analysis introduced in Section 2.6 produces three Level-1 figures, which together support a single, partly counter-intuitive conclusion: the prompt-based behavioral cloning applied in Phase 2 shifts the agents toward a more *scripted and efficient* reasoning style, gaining in adaptation and imitation while losing some of the contingent reasoning observed in the zero-shot baseline.

**Grounding score by turn position.** At turn 0 the grounding score is zero in both phases, as expected (the seller's opening move has no prior dialogue to anchor to). The most diagnostic point is turn 1, the buyer's first move: the Phase 2 grounding score spikes to approximately 0.30, double the Phase 1 value of approximately 0.15. This is the clearest evidence that the cloning is effective — the Phase 2 buyer, primed with an explicit tactic, immediately references the anchors the seller has just introduced (the asking price, the product name, technical terms), whereas the zero-shot Phase 1 buyer responds more generically. At turn 2 the curves cross: the Phase 1 seller, reasoning from scratch about the buyer's bug question, re-references prior content and peaks at approximately 0.23, while the Phase 2 seller executes a planned move and reacts less. From turn 3 onward both curves converge to roughly 0.08–0.10 and decline slowly. We note that part of this decline is a mechanical artefact of the metric: the grounding score divides anchors found by anchors *available*, and the available-anchor pool grows cumulatively each turn while the agent reasons predominantly over recent anchors. Interpretation therefore concentrates on the first three turns, where the signal is cleanest. The Phase 2 curve terminates at turn 6 because Phase 2 negotiations are substantially shorter (mean of approximately 3.9 turns versus 8.5 in Phase 1); comparisons beyond turn 6 reflect Phase 1 alone.

**Conditional specificity.** Across all four configurations, the share of conditional clauses tied to a dialogue anchor is *higher* in Phase 1 than in Phase 2 (approximately 0.47 to 0.30 for configuration A, 0.38 to 0.27 for B, 0.37 to 0.08 for C, and 0.37 to 0.24 for D). This is counter-intuitive — one might expect that learning from successful peers would make reasoning more context-dependent — but it admits a coherent explanation. In the zero-shot baseline, lacking a prescribed tactic, agents reason contingently ("if the buyer accepts I concede, otherwise I hold firm"), and these conditionals naturally reference concrete dialogue anchors. In Phase 2 the tactic is already supplied in the prompt, so reasoning becomes directive ("execute the planned move") rather than conditional, and the conditionals that remain tend to be unanchored boilerplate ("if I push too hard"). 

**CoT length by turn position.** The Phase 2 chain-of-thought is markedly longer at the opening turn (approximately 950 characters versus 400 in Phase 1), reflecting the additional cognitive load of assimilating the injected playbook and mapping it onto the specific scenario before producing the first message. Phase 1, by contrast, displays a bimodal length profile, with peaks at turn 2 (where the buyer typically raises the bug) and again at turns 7–10 (the closing price end-game), followed by a decline. This two-peak signature aligns with the behaviour expected of an agent that deliberates most heavily at the genuine decision points of the negotiation; Phase 2 flattens this profile because the strategic plan is largely pre-determined, front-loading the reasoning at the opening. An isolated Phase 2 spike at turn 5 sits on a wide confidence band and reflects the small number of sessions reaching that depth; it is treated as sampling noise rather than signal.

**Level-2 LLM classifier.** As a complement to the offline Level-1 metrics, an auxiliary `deepseek-chat` classifier rated every turn's chain-of-thought on two 1–3 dimensions — context specificity and planning depth — over the full corpus (759 Phase 1 turns and 367 Phase 2 turns, with no parsing failures); see `cot_llm_scatter.png` and `cot_llm_by_config.png`. Two observations stand out. First, **context specificity edges down slightly from Phase 1 to Phase 2** (mean 2.20 → 2.16 overall), corroborating the Level-1 conditional-specificity finding through an independent method, with configuration D showing the clearest drop (2.33 → 2.11). Second, **planning depth is uniformly low and essentially unchanged** (mean ≈1.61 in both phases): across both conditions the agents rarely engage in explicit multi-branch, "if-X-then-Y-else-Z" planning, tending instead toward single-step or single-contingency reasoning. The convergence of the two methods — a regex-and-anchor offline measure and an LLM judgement — on the same direction of effect strengthens confidence that the reduction in contingent reasoning under social learning is real rather than an artefact of either metric. On the scatter of per-session means, the bulk of sessions fall in the moderate-specificity, low-planning region rather than the high-specificity/high-planning "genuine-reasoning" quadrant, in both phases.

**Synthesis.** Read through the three-lens framework, the picture is consistent: Phase 2 improves on *pragmatic adaptation* (a 100% agreement rate and a higher LLM-judge score across all configurations) and reuses the injected examples enough to register as partial *scripted imitation* (cosine similarity at the calibrated threshold), but shows a modest reduction in the structural correlates of *genuine reasoning* (lower conditional specificity, lower Level-2 context specificity, and a flatter CoT-length profile). This trade-off — efficiency and reliability purchased at the cost of contingent deliberation — is precisely the kind of distinction the analytical framework was designed to expose, and it is revisited in Section 4.

#### Tactic taxonomy

The two tactics extracted by the Observer from Phase 1 and injected as few-shot guidance in Phase 2 represent the dominant winning strategy for each role. Both revolve around the bug as the central negotiation lever, but from opposite sides.

**Seller — Proactive Framed Disclosure.** Rather than concealing or deflecting the known defect, the winning seller tactic is to surface it voluntarily and early, embedding it immediately in a remediation frame (a scheduled fix, a workaround, priority support). This pre-empts the buyer's discovery move, controls the narrative around the bug's severity, and justifies a pre-packaged price concession that still preserves deal value. The tactic is most legible in configurations C and D, where a Fairness-seeker on either side creates the social conditions for transparency to be reciprocated rather than exploited. Example: *"To be fully transparent, there is a known issue in our report export module [...] We have identified the root cause and are rolling out a fix in the next update [...] I can lower the price to 9,000 EUR — a 25% discount."*

**Buyer — Probing + Risk-Anchored Offer.** The winning buyer tactic opens with a targeted interrogation for specific, documented issues rather than generic quality concerns, forcing the seller into a corner where evasion becomes costly. Once a defect is confirmed, the buyer reframes the entire valuation ("data corruption in any module is unacceptable for a production environment") and anchors a low offer as the *maximum* price warranted by the newly revealed risk, pairing it with a concrete condition (written guarantee + fix deadline) that signals seriousness and limits the seller's room to manoeuvre. Example: *"Can you give me concrete examples of known issues or bugs that have been reported in production? [...] I need a price that truly reflects that risk. Can you offer 7,500 EUR, and include a written guarantee on the timeline for the fix?"*

Notably, the two tactics are structurally complementary: the seller's proactive disclosure anticipates and partially defuses the buyer's probing move, which explains the Phase 2 convergence to a narrow price band — both agents arrive at the bug-disclosure juncture faster and with less friction, collapsing the negotiation into three to four turns.

#### Novel tactic emergence (Phase 2)

A manual scan of the 80 Phase 2 transcripts — cross-referenced against the two learned-tactic descriptions in `learned_tactics.json` — reveals that the vast majority of sessions follow the injected playbook closely. Novel moves do surface, but only in sessions that run longer than the Phase 2 mean (6 turns vs. ~3.9), suggesting they emerge as adaptive responses when the standard tactic fails to close quickly rather than as planned strategies. Three deviations are worth noting.

**Seller deflection before disclosure.** In one configuration-B session (`318c18b1`), the seller departs from the proactive-disclosure tactic and initially deflects: when the buyer asks for known issues, the seller responds vaguely ("any minor issues are handled through regular updates") without revealing the bug. Only after the buyer pushes a second time does the seller disclose. This is the behavioural opposite of the injected tactic and represents a genuine improvisation under pressure: *"You're right to push for specifics. To be fully transparent, there is a known issue..."*

**Buyer second-order probing.** In two sessions (`0b83c46d`, config D; `cbb863fa`, config B), after the seller discloses the bug the buyer does not move directly to price anchoring but escalates with a follow-up demand for full disclosure of *any additional* hidden issues. This move is absent from the injected buyer tactic, which focuses on monetising the first discovered bug: *"Before I consider this further, are there any other known bugs or unresolved issues in the suite I should be aware of? I need full disclosure to assess the real value."*

**Buyer walk-away threat as closing lever.** In one configuration-D session (`f4e2e286`), the buyer appends an explicit BATNA signal to their final offer — a move not described in the injected tactic, which frames the anchored price as the maximum justified by risk without threatening exit: *"That's my final offer — if that's not possible, I'll have to walk away and look at other options."*

Taken together, these observations are consistent with a partial scripted-imitation reading: novel tactics are rare, localised in longer sessions, and appear precisely where the injected script runs out of prescriptions — reinforcing the conditional-specificity finding that Phase 2 agents reason contingently only when the pre-supplied plan does not resolve the situation.


### 3.4 Analysis Tooling and Reproducibility

All evaluation logic is implemented as importable Python modules under `src/evaluation/`. `quantitative.py` computes session-level numerical metrics (agreement, price, turns, concessions, linguistic complexity) entirely offline. `qualitative.py` provides four independent components, each aligned with one lens of the analytical framework introduced in Section 2.6: an LLM-judge that scores per-turn appropriateness (pragmatic adaptation), a cosine-similarity routine over sentence embeddings that compares Phase 2 transcripts against the few-shot examples injected in their prompts (scripted imitation), a *per-turn* offline analyser of the private CoT log (`analyse_cot`) that computes the **dialogue grounding score** and the **conditional specificity** of each turn against anchor tokens drawn from the visible dialogue, and a `CoTClassifier` that ships each CoT to `deepseek-chat` for a 1–3 rating on **context specificity** and **planning depth** (genuine reasoning, Level 2). The CoT pipeline replaces an earlier flat regex score that proved non-discriminative because boilerplate phrases (`"I need to ..."`) fired on every turn and saturated the metric across all configurations.

Three Jupyter notebooks orchestrate these modules and produce the figures and tables used in this report. `notebooks/01_baseline_demo.ipynb` and `notebooks/02_social_learning_demo.ipynb` are pedagogical walkthroughs of a single Phase 1 and Phase 2 session respectively, intended for the demonstration component of the assessment. `notebooks/03_analysis_visualization.ipynb` is the analytical core: it loads both phases from the raw JSON transcripts, builds the aggregated comparison table reproduced in §3.2 and §3.3, and renders all figures into `data/results/figures/`. Each API-dependent analysis (the LLM-judge, the Level-2 chain-of-thought classifier) and the cosine-similarity computation cache their output to `data/processed/` — as `<run_id>_judge.json`, `<run_id>_cot_llm.json`, and the model weights respectively — so the notebook can be re-opened without re-incurring API or model-loading costs. This separation between modular library code and notebook-driven presentation follows the course's explicit guidance against monolithic notebooks.

---

## 4. Concluding Remarks

### 4.1 Summary of Findings

The study set out to ask not merely *whether* LLM agents negotiate well, but *what kind* of competence underlies their behaviour. The two-phase design, read through the three-lens framework, gives a layered answer.

The most salient result is a **trade-off surfaced by the framework rather than by any single metric.** Prompt-based behavioral cloning (Phase 2) makes the agents demonstrably more effective negotiators: agreement rose from 70% to 100% across configurations, negotiations closed in roughly half the turns, and prices converged tightly toward the seller's reservation value (configuration C settling at exactly €7,500 in every session). The pragmatic-adaptation lens agrees — the LLM-judge's appropriateness score rose from ≈2.65 to ≈2.96. Yet the genuine-reasoning lens tells a more cautionary story: the chain-of-thought analysis shows that this efficiency is accompanied by a shift *away* from contingent, dialogue-anchored reasoning toward more directive, script-following reasoning. Agents ground their reasoning in the dialogue faster (a higher grounding score at the first response) but deliberate less conditionally — conditional specificity falls in all four configurations, the Level-2 context-specificity score edges down through an independent method, and the per-turn effort profile flattens, losing the bimodal "deliberate at the decision points" signature of the zero-shot baseline. The scripted-imitation lens is consistent with this: Phase 2 transcripts sit right at the calibrated cosine-similarity threshold, reusing the gist of the injected examples while varying the surface form.

Taken together, the lenses converge on a single interpretation: **prompt-based social learning buys efficiency and reliability at the cost of some genuine, contingent reasoning.** This is not a defect of the method but a clarifying observation about what "social learning via prompting" actually does to an agent's internal process — It replaces nuanced, context-dependent reasoning with a pre-validated, standardized script. Crucially, an evaluation relying only on outcome metrics (agreement rate, price, turns) would have recorded an unambiguous success and missed this entirely; it is the combination of lenses, and in particular the access to the private chain-of-thought, that makes the trade-off visible.

### 4.2 Limitations

Several limitations qualify these conclusions and should be read as caveats on the findings above.

- **Single-model dependence.** Every agent, judge, and classifier is a DeepSeek model. The genuine-reasoning analysis is only possible because DeepSeek-R1 exposes its chain-of-thought, but this also means the findings may not transfer to other model families, and the chain-of-thought itself is a *rationalisation produced by the model*, not a guaranteed faithful trace of its computation.
- **LLM-evaluating-LLM circularity.** The Observer, the appropriateness judge, and the Level-2 CoT classifier are themselves LLMs. The only human verification performed was the cosine-similarity threshold calibration (six manually labelled sessions); no human annotation was applied to the LLM-judge scores or the CoT classifier outputs, so those results should be read as indicative.
- **Keyword-based bug detection.** Disclosure and discovery flags are set by keyword matching, which measures whether the *topic* surfaced rather than making a precise semantic judgement; the near-ceiling rates should be read in that light. The keyword list includes broad terms (`"concern"`, `"risk"`, `"worry"`) that can fire on ordinary negotiation language even when the bug is not explicitly at issue, potentially inflating both `bug_disclosed` and `bug_discovered` counts. A semantic classifier — for instance an LLM prompt asking "does this turn specifically reference the known software defect?" — would yield a more precise signal and is noted as a direction for future work.
- **Cosine-similarity metric.** As the calibration exercise showed, sentence-embedding similarity captures semantic but not verbatim lexical overlap, and the 0.75 threshold is a human-calibrated heuristic from a small sample. A complementary n-gram (ROUGE-n) measure would strengthen the imitation lens.
- **Grounding-score artefact.** The dialogue-grounding score has a denominator (available anchors) that grows cumulatively over a negotiation, so its decline across turns is partly mechanical; interpretation is therefore concentrated on the early turns, and a recency-weighted variant is left for future work.

- **Scope.** A single scenario, one model, four persona pairings, and 20 sessions per cell limit statistical power and external validity; no out-of-distribution perturbation tests (different product, bug type, or price range) were run, although the framework is designed to accommodate them.

### 4.3 Potential Extensions and Future Work

Several directions could extend this work beyond the current scope:

- **LoRA-based Behavioral Cloning:** The in-context approach used here is a functional approximation of behavioral cloning. A natural extension would be to use LoRA (Low-Rank Adaptation) fine-tuning on an open-weight model (e.g. LLaMA 3, Mistral) to implement true gradient-based BC, training on the Phase 1 success demonstrations. This would allow for a direct comparison between prompt-based and weight-update-based cloning in terms of behavioural generalisation and robustness. Ruled out for the current project due to time and resource constraints, but a straightforward next step.
- **Out-of-distribution perturbation tests:** the framework described was designed to accommodate controlled context changes (different software product, different bug type, different price range, reduced budget, time pressure), but these tests were not run within the current project scope. Running them would allow a sharper distinction between scripted imitation (invariant phrasing despite changed context) and genuine adaptation (coherent strategy adjustment), and would strengthen the external validity of the findings.
- **Multi-round iterated learning:** running multiple Phase 2 cycles to observe whether strategies converge, diverge, or oscillate over generations.
- **Multimodal agents:** introducing structured data (product spec sheets, financial tables) into the negotiation context.
- **Human-in-the-loop:** replacing one agent with a human participant to study human–LLM negotiation dynamics.
- **Recency-weighted grounding score:** the current dialogue-grounding metric divides anchors found by all anchors available, so the denominator grows cumulatively and mechanically depresses later-turn scores. A recency-weighted variant — giving higher weight to anchors introduced in the most recent turns — would produce a fairer comparison across turn positions.
- **Semantic bug-disclosure detection:** replacing the current keyword-based `bug_disclosed` / `bug_discovered` flags with an LLM-based classifier that determines whether a given turn *specifically references the known software defect*, rather than merely containing related vocabulary. This would allow for a finer-grained analysis of *how* the information asymmetry resolves — distinguishing strategic framed disclosure from incidental mention — and would eliminate the inflation in near-ceiling rates caused by broad trigger terms such as `"concern"` or `"risk"`.

---
### Relevant Literature

This work sits at the intersection of three strands of research: end-to-end learning of negotiation dialogue, emergent and social behaviour in multi-agent LLM systems, and reasoning-exposing language models.

**1. Lewis et al. (2017), *Deal or No Deal? End-to-End Learning of Negotiation Dialogues.*** This is the foundational work on training neural models to negotiate via natural language. Using a multi-issue bargaining task, the authors showed that supervised learning on human dialogues, refined with self-play and reinforcement learning, could produce agents that negotiate end-to-end — and, notably, that such agents spontaneously learned to deceive (feigning interest in an item only to concede it later for advantage). Our project inherits the central premise (that negotiation is a productive task for studying emergent communicative strategy) but differs in two important ways: we use a pre-trained instruction-following LLM rather than a model trained from scratch on the task, and our analytical emphasis is not on outcome optimisation but on *characterising the cognitive status* of the strategies that appear.

**2. Gupta et al. (2025), *The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems.*** This work studies how cooperative norms emerge in populations of LLM agents through social learning — agents observing and adapting to the behaviour of successful peers. It motivates our Phase 2 design: rather than fine-tuning, we implement social learning *in-context*, having an Observer agent identify the most successful negotiation tactics and reinject them into subsequent agents' prompts. Our use of the term "social learning" follows this prompt-engineering sense rather than a strict reinforcement-learning one, a distinction we make explicit in the methodology.

**3. Akin et al. (2025), *Socialized Learning and Emergent Behaviors in Multi-Agent Systems based on Multimodal LLMs.*** This paper examines emergent behaviours that arise when multimodal LLM agents interact and learn from one another in a shared environment. It provides conceptual grounding for the idea that interaction itself — not just the base model — shapes agent behaviour, and that emergent strategies can be studied as a phenomenon in their own right. We draw on this framing to treat the strategies that surface in our negotiations (and any *novel* tactics that appear only after social learning) as objects of analysis, while restricting our own setting to the unimodal, text-only case.

**4. DeepSeek-AI (2025), *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.*** DeepSeek-R1 is the reasoning model we use throughout. Its defining property for our purposes is that it returns its chain-of-thought in a separate `reasoning_content` field rather than embedding it invisibly. This is what makes the *genuine reasoning* lens empirically tractable: we can inspect the private deliberation behind each move. By contrast, OpenAI's o-series models generate reasoning tokens internally but do not expose them via the API, which would make the same analysis impossible. The methodological dependence of our study on this single capability is worth stating plainly, and we return to its limitations in the concluding discussion.

---


## AI Usage Disclaimer

The following AI assistants were used in the course of this project: Claude Sonnet 4.6 (Anthropic) and Gemini 3 (Google).
Both models were used for: brainstorming research directions and experimental design; writing, debugging, and reviewing code across the src/ modules and Jupyter notebooks; locating and summarising relevant academic papers; supporting the design of the project architecture (module structure, data flow, evaluation pipeline); and drafting and editing sections of this report.
All research questions, experimental choices, analytical interpretations, and final conclusions are the author's own. AI-generated code was read, tested, and validated by the author before use. AI-generated prose was reviewed and edited by the author. The author retains full responsibility for the content, reasoning, and conclusions of this work.