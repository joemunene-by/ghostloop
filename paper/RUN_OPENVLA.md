# Reproducing OpenVLA-7B / π0 / Octo through ghostloop's bench

This is the playbook for the follow-up paper — running an actual
published VLA checkpoint through `ghostloop.bench.VLABenchmarkSuite`
and publishing the comparison numbers under our safety pipeline.

## Why it matters

The first ghostloop preprint positions the framework. The follow-up
positions the *measurements*. Running OpenVLA-7B (or π0, or Octo) and
showing concrete pass-rate gaps under the safety pipeline turns
ghostloop from "framework that could measure VLAs" into "framework
that has measured VLAs" — different reputation tier.

Three benches ship pre-populated with published baselines:

| Bench | Models with published numbers | Citation |
|---|---|---|
| `pick_place_widowx` | OpenVLA-7B (57.5%), RT-2-X (49.5%), Octo-Base (53.5%) | Kim 2024 / Brohan 2023 / Octo 2024 |
| `manipulation_bridge` | π0 (65%), OpenVLA-7B fine-tuned (45%) | Black 2024 / Kim 2024 |
| `reach_target` | Diffusion Policy (78%), ACT (72%) | Chi 2023 / Zhao 2023 |

## Hardware needed

| GPU | What it runs |
|---|---|
| 1× A100 80GB | OpenVLA-7B FP16, full speed (~2-3s/step) |
| 1× A10 24GB | OpenVLA-7B INT8 via bitsandbytes (~5-7s/step) |
| 1× H100 80GB | OpenVLA-7B FP16 fastest (~1-2s/step) |
| 1× RTX 4090 | π0 (lighter); INT8 OpenVLA possible but slow |

CPU-only: not viable. A 7B-param VLA is too slow per step.

Cheapest path: rent A10 from RunPod / Lambda / Vast.ai for ~$0.40/hr.
20-episode bench runs in ~30 min; full reproduction (n=200) in ~5 hr.
~$2 of GPU time per data point.

## Software setup

```bash
# Provision the machine. Ubuntu 22.04 / CUDA 12.x.
sudo apt update && sudo apt install -y python3-venv git
python3 -m venv ~/.venvs/ghostloop && source ~/.venvs/ghostloop/bin/activate

# ghostloop with the relevant extras.
pip install --upgrade pip
pip install ghostloop[mujoco,gym,mcp]

# OpenVLA dependencies.
pip install transformers>=4.40 accelerate torch
pip install bitsandbytes      # only if INT8

# Sim env. WidowX in MuJoCo:
pip install gymnasium-robotics
# OR Bridge-style envs:
pip install bridge-data-robot-suite     # check with the OpenVLA team
```

## Run the reproduction

```bash
# Easiest: use the script ghostloop ships.
GHOSTLOOP_OPENVLA_CKPT=openvla/openvla-7b \
GHOSTLOOP_BENCH=pick_place_widowx \
GHOSTLOOP_N_EPISODES=20 \
GHOSTLOOP_OUTPUT_DIR=./runs/openvla_$(date +%Y%m%d_%H%M%S) \
GHOSTLOOP_DEVICE=cuda \
GHOSTLOOP_DTYPE=fp16 \
    python3 examples/openvla_reproduction.py
```

The script does:
1. Load OpenVLA-7B onto the GPU.
2. Build N Episodes targeting `pick_place_widowx`.
3. Run each episode: at every step, render the camera frame, prompt
   OpenVLA with the goal, get a 7-D action vector, map it to a
   ghostloop `Intent`, run through the safety pipeline, dispatch.
4. Score each episode against the success predicate.
5. Compute Cohen's h vs every published baseline + render a Markdown
   report.
6. Save per-episode trace JSONL files for downstream property mining
   / counterfactual replay / LLM-as-judge scoring.

## Wiring the real OpenVLA path

The shipped script has a stub policy by design — letting you validate
the harness end-to-end on the dev box before paying for GPU time.
Two edits replace the stub with real inference:

**In `examples/openvla_reproduction.py`'s `OpenVLAPolicyAdapter.act`:**

Uncomment the real-OpenVLA block at the bottom of the method, then
adjust:
- `unnorm_key`: the OpenVLA-specific stat key for your sim env
  (`"bridge_orig"` for WidowX, others for different envs).
- The (dx, dy, dz) → `move_to` mapping: the action is a *delta*; add
  it to the current end-effector position.

**In `_episode_for_widowx_pick_place`'s `setup()`:**

Replace `MockBackend()` with the real sim backend:

```python
from ghostloop.backends import MuJoCoBackend
return MuJoCoBackend(
    model_path=str(WIDOWX_XML_PATH),  # WidowX MJCF you've installed
    end_effector="gripper",
)
# OR Gymnasium:
from ghostloop.backends import GymnasiumBackend
return GymnasiumBackend(env_id="WidowXPickPlace-v0")
```

And replace the empty `primitives=lambda: []` with the real arm
primitives bound to that backend.

## What to expect

OpenVLA's published WidowX number is 57.5% pass on n=200 (paper
Table 4). On a smaller sample (n=20-50) you'll see noisier numbers
— Wilson 95% CI is wider — but the central tendency should land
within ±15% of the paper's value if everything is wired right.

Common gotchas:
- `unnorm_key` mismatch → action vector at the wrong scale → 0%
  pass rate. Triple-check the key matches your sim env.
- Camera frame format (RGB vs BGR, 0-255 vs 0-1) → garbled inputs.
  OpenVLA expects RGB uint8 256×256.
- Action coordinate frame (world vs end-effector) → arm flails.
  WidowX uses end-effector frame in OpenVLA's defaults.

## What the safety pipeline adds

Crucially: ghostloop's `pick_place_widowx` config defaults the
`GeofenceGate` + `ForceCapGate` + `ActionSmoothingGate`. So the
numbers you publish are NOT directly comparable to OpenVLA's paper
— OpenVLA had no such constraints. **That's the point of the paper:**
report the gap. *"Under our 12-gate safety pipeline, OpenVLA-7B
hits 53.0% on pick_place_widowx (Wilson CI [42, 64]), down 4.5
points from the unconstrained 57.5% baseline. Most of the loss
comes from the GeofenceGate rejecting reaches outside the
calibrated workspace; ablating that gate recovers 56.8%."*

That's a publishable result.

## After the run

The output directory contains:
- `report.md` — the human-readable Cohen's h table
- `report.json` — machine-readable summary for CI / dashboards
- `traces/*.jsonl` — one trace per episode, ready for:
  - **Property mining**: `mine_properties(traces)` — auto-discover
    the invariants OpenVLA naturally satisfies vs the ones it breaks.
  - **LLM-as-judge**: `LLMJudge(client).score_many(traces)` — qualitative
    rubric scoring of the 20 episodes for the appendix.
  - **Counterfactual replay**: `replay_with_policy(trace, scripted_baseline)`
    — what would a hand-written policy have done at each step?
- `safety_report.md` — `PropertyEngine` evaluation across the corpus.

These four artifacts together comprise the methodology section of
the follow-up paper.

## Suggested venues

- **Workshop submissions** (low bar to entry):
  - ICRA RoboTrust workshop
  - NeurIPS Robot Learning workshop
  - CoRL Embodied Decisions workshop

- **Main track** (need n=200+ + ablations):
  - CoRL 2026
  - ICRA 2027

The right move is workshop first, get feedback, then expand to main
track with the ablations + the SoftGate variants.
