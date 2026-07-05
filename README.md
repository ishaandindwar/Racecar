# Racecar — RL Self-Driving Agent

A reinforcement learning agent trained with PPO (via Stable-Baselines3) to
drive on Gymnasium's `CarRacing-v3` track. Built as part of ongoing
self-directed learning in reinforcement learning and computer vision.

## Branches

**`main`** — the original, simple version. A single `CarRacing-v3`
environment, PPO with default hyperparameters, no obstacles. This is
kept intentionally untouched.

**`obstacle-experiment`** — an active experiment branch layering two things
on top of the baseline:
1. Static obstacles physically embedded in the track (a genuine Box2D
   environment change, not just a wrapper).
2. A corrected training setup — the first version of this experiment
   revealed a real RL failure mode, which led to fixing the
   exploration strategy and adding an early-termination wrapper.

This branch is not yet merged into `main`. It stays separate on purpose
until it's been trained long enough to confirm it's a genuine improvement,
not just a different set of choices.

## What's in `obstacle-experiment`

| File | What it does |
|---|---|
| `obstacle_env.py` | Subclasses Gymnasium's `CarRacing` to add static Box2D obstacle bodies onto the track, with real physical collision and a reward penalty on contact. |
| `racecar_env.py` | Builds either the plain or obstacle-augmented environment, backward-compatible with the original (obstacles are off by default). |
| `train.py` | PPO training loop: parallel environments, action-repeat, State-Dependent Exploration, and an early-termination wrapper for stalled episodes. |
| `play.py` | Loads a trained model and renders it driving. |

## This projects focus: understanding *why*

Following on from RL fundamentals (the training loop, PPO vs.
DQN vs. SAC, gymnasium/stable-baselines3 basics), I went deeper
into diagnosing *why* a trained agent behaves the way it does, rather than
just accepting a reward number at face value.

**What I explored:**
- Box2D physics fundamentals: static vs. dynamic bodies, fixtures, and how
  contact detection actually works under the hood (including a real bug I
  hit — Box2D returns a new Python proxy object every time you access a
  body through a contact, so plain attributes don't survive; you have to
  use the `userData` field instead, which preserves object identity).
- PPO's clipped surrogate objective — why "proximal" specifically means
  capping how far one update can move the policy, and why that matters for
  training stability.
- On-policy vs. off-policy learning, and why that distinction is the reason
  parallel environments (`SubprocVecEnv`) matter for PPO specifically —
  on-policy algorithms can't reuse old rollout data the way DQN can from a
  replay buffer.
- Exploration strategy for continuous action spaces: why an entropy bonus
  (good for discrete actions) produces jittery, uncorrelated noise for
  continuous steering, and why State-Dependent Exploration (holding a
  sampled noise function for several steps instead of resampling every
  step) is the better fit here.


**What I found :**

After training the obstacle-augmented agent for 300k timesteps, the agent
had converged to a degenerate policy: sit still on the grass and let the
small per-frame penalty (`-0.1`) accumulate, rather than risk driving. The
reward numbers (around -150 to -170 per episode) matched almost exactly
what you'd expect from ~1600-1700 steps of pure time penalty with zero
tile progress — confirming this wasn't a training bug, it was the agent
correctly optimizing an objective it hadn't yet learned to solve properly.




Two real causes, both conceptual rather than "just add more compute":
1. **Reward landscape**: with too little experience, the
   guaranteed small loss of standing still beat the uncertain, riskier path
   toward learning to drive (which risks a much larger -100 penalty for
   leaving the map). This is a credit-assignment problem.
2. **A termination-boundary mismatch**: the environment only ends an
   episode when the car strays ~333 units from center (`PLAYFIELD`), while
   the road itself is only ~7 units wide (`TRACK_WIDTH`). So "off the road"
   and "episode over" are very different thresholds — the agent could sit
   on grass for the entire episode length without the environment ever
   stepping in.

**Fixes applied** (in the current `train.py` on this branch):

- Added an `OffTrackTimeout` wrapper that ends an episode early if no new
  road-tile reward has been earned for a set number of consecutive steps,
  so training time isn't spent on hopeless rollouts.

**Still open:** the corrected setup hasn't been trained to convergence yet
(that needs a long training time — CarRacing typically needs millions of timesteps
for a good policy, well beyond what a quick verification run shows). The
priority of this project was confirming the diagnosis and fixing the actual
cause, not burning compute on a training run before understanding what
was wrong.

## Running it

```bash
pip install -r requirements.txt

# baseline (main branch)
python train.py --timesteps 100000
python play.py

# obstacle-experiment branch
python train.py --timesteps 300000 --n-envs 4 --num-obstacles 6
python play.py --num-obstacles 6 --episodes 5
```
