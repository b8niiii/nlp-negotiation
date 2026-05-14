# CLAUDE.md — NLP Project Reference

> This file is the primary reference for the AI assistant working on this project.
> It must be read at the start of every session. It defines project scope, rules, and guidelines.

---

## Project Overview

**Title:** LLM Agents in Negotiation: Emergent Communicative Strategies in Multi-Agent Dialogue

**Course:** NLP (University)
**Student:** Alessandro Bottoni

**Core Idea:**
Investigate how Large Language Models behave as autonomous agents engaged in negotiation, cooperation, or strategic dialogue. Two or more models are placed in simulated scenarios where they must reach an agreement, trade resources, or align on decisions despite having distinct goals or incomplete information. The objective is to analyze emergent communicative strategies—persuasion, concession, deception, cooperation—and to evaluate whether these behaviors reflect genuine reasoning, pragmatic adaptation, or scripted imitation.

---

## Methodology (must follow this structure)

### 1. Scenario Design
- Define one or more negotiation settings:
  - **Resource division** — splitting items or money
  - **Task scheduling** — allocating responsibilities
  - **Preference alignment** — choosing the best option for both parties
- Each agent receives private information or asymmetric incentives encoded in its prompt.

### 2. Agent Configuration
- Instantiate two or more LLM agents with distinct "personas" or objectives.
  - Agent A: seeks maximum profit
  - Agent B: values fairness
  - Agent C: minimizes risk
- Optionally include an **adjudicator model** (or human evaluator) to judge outcomes.

### 3. Dialogue Simulation
- Iterative rounds of conversation until agreement or impasse.
- Test variations:
  - **Cooperative mode** — shared goal
  - **Competitive mode** — conflicting goals
  - **Mixed mode** — partial cooperation or deception allowed

### 4. Analysis & Metrics
**Quantitative:**
- Agreement rate
- Rounds to convergence
- Utility scores
- Language complexity

**Qualitative:**
- Persuasion tactics
- Emotional tone
- Logical coherence of dialogue transcripts

---

## Dataset
No fixed dataset required. Negotiation scenarios can be:
- Synthetically generated
- Adapted from existing dialogue datasets

---

## Expected Outcomes
- Uncover how cooperative or adversarial behaviours emerge among LLMs
- Identify pragmatic and linguistic features correlated with success or failure in negotiation

---

## Key References

- Lewis, M., Yarats, D., Dauphin, Y., Parikh, D., & Batra, D. (2017). *Deal or No Deal? End-to-End Learning of Negotiation Dialogues.* EMNLP 2017.
- Akin, S. et al. (2025). *Socialized Learning and Emergent Behaviors in Multi-Agent Systems based on Multimodal LLMs.* arXiv:2510.18515.
- Gupta, P. et al. (2025). *The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems.* arXiv:2510.14401.

---

## Course Requirements (always respect these)

### Report Structure (4 sections)
1. **Introduction** — Overview + pertinent literature
2. **Research Question & Methodology** — Goals, approach, formal problem definition
3. **Experimental Results** — Dataset, metrics, methodology, results
4. **Concluding Remarks** — Critical discussion + future work

### AI Usage Policy
- AI tools may be used for support, but must NOT substitute original work
- A mandatory disclaimer must appear in the final report specifying which models were used, for what purposes, and how outputs were modified/verified

### Coding Standards
- Code must be modular and well-structured (no single large notebook)
- Use Python modules + OOP for logic
- Use Jupyter only for demo/experimentation/visualisation
- Separate concerns: data loading, preprocessing, model interface, evaluation, visualisation
- Maintain a clean, reproducible, extensible repository

---

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | This file — project reference for the AI assistant |
| `project_notes.md` | Living tracker: progress, to-dos, decisions, log |
| `presentation.md` | Evolving project report / paper |

---

## Persistent Rules for the AI Assistant

1. **Always read `CLAUDE.md` and `project_notes.md` at the start of each session.**
2. **Always update `project_notes.md`** after every meaningful task (new decisions, progress, completed to-dos).
3. **Always update `presentation.md`** when working on report content.
4. The methodology described above must be respected throughout the project.
5. All code must follow the modularity and separation-of-concerns principles above.
