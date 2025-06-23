## Network quick-facts *(today’s version)*

- Two fully-connected **LIF** layers (64 neurons each)
- Shared weights for **actor / critic**
- No recurrence, no skip connections, single read-out linear layer
- Fixed β = 0.9 leak, identical time-constants across neurons
- Only *rate / temporal-average* decoding

---

### Why this is a good baseline
This is basically the SNN analogue of a 2-layer MLP.  Capacity ↔ convergence in spiking RL is trickier than in ANNs; scaling must keep spike statistics healthy.

---

## Playbook for scaling the network

### 1. Validate current width first
You know it works when:

- `spike_ratio` ≈ **0.25–0.40**
- entropy doesn’t collapse early
- returns plateau at a sensible value

Only **then** start changing size.

### 2. If the policy **under-fits** (score < ANN)

a. **Width**: `64 → 128` (both hidden layers). Keep LR, β, `num_steps` identical; watch `spike_ratio`.

b. **Depth**: add a **third LIF layer (64)**. Bigger/deeper usually needs more timesteps (e.g. 96–128).

**Symptoms of too much capacity**

- entropy plunges to *0* quickly
- `spike_ratio` > 0.6
- weight-change metrics explode despite grad-clipping

### 3. If the policy **over-fits / unstable**

- Narrow to **32** units per layer
- Add dropout on the read-out rate: `nn.Dropout(p=0.3)` after `avg_spk2`
- Increase weight decay, e.g. `Adam(..., weight_decay=1e-4)`

### 4. Architectural upgrades that often help

- Separate **actor / critic** networks (different init + dropout)
- **Heterogeneous** neuron params: draw β ∈ U(0.8, 0.95)
- Spike-frequency adaptation (`snn.AdaptiveLIF`)
- Simple recurrent head: feed last layer spikes back (small α weight)
- Residual current: `x → fc → LIF` **+** skip of raw current

### 5. Encoding alternatives (often bigger gains than width)

Rate + 32 steps is wasteful.  Try:

- **Latency code** (time-to-first-spike) with 64 steps – net size unchanged
- **Population code**: random Gaussian projection → higher-dim spike input, smaller hidden size

---

## Practical hyper-parameter grid (ascending complexity)

| hidden | depth | num_steps | notes |
|-------:|------:|-----------:|-------|
| 64 | 2 | 64 | Baseline after bug-fixes |
| 128 | 2 | 64 | More width |
| 64 | 3 | 96 | More depth |
| 32 | 2 | 64 | Smaller, faster – good for ablation |
| 64 | 2 | 64 | β heterogeneous, SFA |

Use the **same total training steps** for fairness. Track: *return, entropy, spike_ratio*.

---

## Monitoring hints

- Keep `avg_spike_time` roughly mid-range (not always 1 or `num_steps`).
- If `clip_fraction` in PPO stays > 0.3 after a few epochs → network too large / LR too high.
- Visualise weight histograms; spiking nets often drift toward all-positive weights – apply weight decay or re-centre periodically.

---

### Bottom line
Fix the encoder/decoder & firing-rate issues first; you might already reach reasonable returns with the **2 × 64** SNN.  Afterwards, scale width (×2) **before** depth, add heterogeneity, and increase `num_steps` only when `spike_ratio` gets too low.