# LLM Agents in Negotiation: Emergent Communicative Strategies in Multi-Agent Dialogue

> **Status:** Work in progress
> **Last updated:** 2026-05-14

---

## Authors

- Alessandro Bottoni

## Abstract

*[To be written — concise summary of research question, methodology, and key findings]*

---

## 1. Introduction

*[To be written — overview of the project and relevant literature]*

### Relevant Literature

*[To be populated — key works to discuss:]*
- *Lewis et al. (2017) — Deal or No Deal? — foundational work on end-to-end negotiation learning*
- *Akin et al. (2025) — Socialized Learning and Emergent Behaviors in Multi-Agent Systems — multi-agent emergence*
- *Gupta et al. (2025) — Social Learning and Collective Norm Formation in LLM Multi-Agent Systems — norm emergence via observation*
- *DeepSeek-AI (2025) — DeepSeek-R1 — reasoning model with exposed chain-of-thought*

---

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

This scenario is designed to elicit a rich range of communicative strategies: strategic omission, bluffing, probing questions, framing, rapport-building, and concession management. The bug serves as a natural information asymmetry pivot around which the negotiation dynamics unfold.

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

### 2.4 Model

All agents are instantiated using **DeepSeek-R1**, a reasoning-capable large language model that exposes its chain-of-thought via a dedicated `reasoning_content` field in the API response, separate from the final `content` field sent to the user. This property is exploited to implement a **private Chain-of-Thought** mechanism: each agent's internal reasoning is logged for offline analysis but is never transmitted to the opposing agent. This provides direct access to the agent's reasoning process without any architectural modification.

OpenAI o-series models (o1, o3) were considered but rejected for this purpose, as their reasoning tokens are generated internally and not exposed via the API.

### 2.5 Experimental Design

The experiment proceeds in two phases.

#### Phase 1 — Baseline Zero-Shot Negotiation

Each configuration (A–D) is run for approximately 20 negotiations, yielding ~80 total baseline runs. Each negotiation proceeds iteratively until either an agreement is reached or an impasse is declared (maximum 15 turns). For each run, the following are logged:

- Full dialogue transcript
- Private CoT per agent turn (from `reasoning_content`)
- Final agreed price or "no deal"
- Whether the bug was disclosed and/or discovered
- Number of turns to outcome

Phase 1 produces baseline distributions for quantitative metrics and raw material for qualitative analysis.

#### Phase 2 — In-Context Social Learning and Prompt-Based Behavioral Cloning

Phase 2 implements a lightweight version of social learning and behavioral cloning entirely through prompt engineering, drawing conceptual inspiration from Gupta et al. (2025) and Akin et al. (2025).

A third DeepSeek-R1 instance acts as an **Observer/Arbitrator**. After Phase 1, the Observer reads all negotiations and scores each agent's performance. For each role, the 1–2 most successful negotiations are selected and the Observer extracts: (a) an abstract natural-language description of the winning tactic, and (b) representative dialogue snippets.

The system prompts of subsequent agents are then updated to include the extracted tactic and snippets as few-shot guidance. This constitutes **prompt-based behavioral cloning**: agents are not fine-tuned, but their in-context behaviour is shaped by exemplary demonstrations. Phase 2 then runs the same ~80 negotiations with updated prompts. The comparison between Phase 1 and Phase 2 outcomes forms the basis for measuring learning and behavioural change.

**Terminological note:** The terms "behavioral cloning" and "social learning" are used here in an extended, prompt-engineering sense rather than their strict machine-learning definitions (which imply gradient-based weight updates). This distinction is made explicit and is justified by the functional analogy: agents learn from observing successful peers and update their behaviour accordingly, even if the mechanism differs from classical RL/IL approaches.

### 2.6 Analysis Framework — Distinguishing Reasoning from Imitation

Each negotiation is analysed through the three-lens framework introduced in Section 2.1. The following operational criteria are used:

**Scripted Imitation:**
- High cosine similarity (sentence-embedding level) between new negotiations and the few-shot examples provided in the prompt.
- Invariant phrasing across out-of-distribution perturbations (different software product, different bug type, different price range) despite changed context.

**Pragmatic Adaptation:**
- Behaviour changes coherently with counterpart responses (e.g. Buyer escalates pressure only when Seller appears evasive).
- Under controlled perturbations (reduced budget, time pressure, generous opening offer), the agent adjusts strategy in a contextually appropriate direction.
- Per-turn LLM-judge score: "Is this move appropriate given the dialogue so far?"

**Genuine Reasoning:**
- CoT analysis: presence of explicit conditional logic ("If I push too hard on the bug, the deal might collapse"), references to prior turns, and if-then-else structures.
- Post-Phase-2 novelty: LLM-judge assessment of whether strategies in Phase 2 include elements not present in the original prompts.
- Cross-episode coherence: consistent but contextually variable policy across runs with similar conditions.

A human inter-rater check is applied to a subset of Observer scores to assess reliability and mitigate circularity (using an LLM to evaluate other LLMs).

---

## 3. Experimental Results

*[To be written — dataset overview, evaluation metrics, experimental methodology, and results]*

### 3.1 Dataset Overview

*[To be written — description of generated negotiation corpus: number of runs per config, total turns, token counts]*

### 3.2 Quantitative Results

*[To be written — tables and figures: price distributions, agreement rates, disclosure rates, turns to convergence, Phase 1 vs Phase 2 comparison]*

### 3.3 Qualitative Analysis

*[To be written — tactic taxonomy, CoT analysis, novel tactic emergence, representative examples]*

---

## 4. Concluding Remarks

*[To be written — critical discussion of results and ideas for future work]*

### Potential Extensions and Future Work

Several directions could extend this work beyond the current scope:

- **LoRA-based Behavioral Cloning:** The in-context approach used here is a functional approximation of behavioral cloning. A natural extension would be to use LoRA (Low-Rank Adaptation) fine-tuning on an open-weight model (e.g. LLaMA 3, Mistral) to implement true gradient-based BC, training on the Phase 1 success demonstrations. This would allow for a direct comparison between prompt-based and weight-update-based cloning in terms of behavioural generalisation and robustness. Ruled out for the current project due to time and resource constraints, but a straightforward next step.
- **Multi-round iterated learning:** running multiple Phase 2 cycles to observe whether strategies converge, diverge, or oscillate over generations.
- **Multimodal agents:** introducing visual or structured data (product spec sheets, financial tables) into the negotiation context.
- **Human-in-the-loop:** replacing one agent with a human participant to study human–LLM negotiation dynamics.

---

## References

- Lewis, M., Yarats, D., Dauphin, Y., Parikh, D., & Batra, D. (2017). *Deal or No Deal? End-to-End Learning of Negotiation Dialogues.* EMNLP 2017.
- Akin, S. et al. (2025). *Socialized Learning and Emergent Behaviors in Multi-Agent Systems based on Multimodal LLMs.* arXiv:2510.18515.
- Gupta, P. et al. (2025). *The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems.* arXiv:2510.14401.
- DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948.

---

## AI Usage Disclaimer

*[To be written before submission — specify which models were used, for what purposes, and how outputs were modified/verified]*
