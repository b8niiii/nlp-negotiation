# LLM Agents in Negotiation
### Emergent Communicative Strategies in Multi-Agent Dialogue

NLP university project — Alessandro Bottoni

---

## Overview

This project investigates what communicative strategies emerge when LLM agents
engage in multi-round negotiation, and whether these reflect genuine reasoning,
pragmatic adaptation, or scripted imitation.

Two DeepSeek-R1 agents (Seller and Buyer) negotiate the sale of a software
product with deliberate information asymmetry: the Seller knows about a
critical bug and must decide whether and how to disclose it.

See `presentation.md` for the full research report and `project_notes.md` for
the living project tracker.

---

## Setup

```bash
# 1. Clone and enter the project
cd NLP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

---

## Running Experiments

```bash
# Phase 1: Baseline zero-shot negotiations (20 runs per config, all 4 configs)
python experiments/run_phase1.py --runs 20

# Phase 2: Social learning with extracted tactics
# First, run Phase 1 and note the run ID printed at the end, then:
python experiments/run_phase2.py --phase1-run-id <run_id> --runs 20
```

---

## Repository Structure

```
NLP/
├── config/             # Scenario and persona YAML configs
├── src/
│   ├── agents/         # BaseAgent, NegotiatingAgent, ObserverAgent
│   ├── simulation/     # DialogueLoop, NegotiationSession
│   ├── social_learning/ # TacticExtractor, PromptUpdater
│   ├── evaluation/     # Quantitative, Qualitative, Metrics
│   ├── logging/        # TranscriptLogger
│   └── utils/          # DeepSeekClient
├── data/
│   ├── raw/            # Full transcripts + CoT logs (JSON)
│   ├── processed/      # Learned tactics, intermediate data
│   └── results/        # Summary CSVs
├── notebooks/          # Demo and analysis notebooks
└── experiments/        # Experiment runner scripts
```

---

## Key Design Decisions

- **Model:** DeepSeek-R1 — chosen for native `reasoning_content` field (private CoT)
- **Private CoT:** agent reasoning is extracted before forwarding messages to the other agent
- **Social learning:** implemented as prompt-based behavioral cloning (in-context, no weight updates)
- **Analysis framework:** scripted imitation / pragmatic adaptation / genuine reasoning
