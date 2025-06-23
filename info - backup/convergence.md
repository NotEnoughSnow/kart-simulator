## Key convergence blockers (ordered by impact → ease of fix)

---

### 1. Encoding / decoding mismatches

- `generate_spike_trains` mixes **NumPy** with a `torch.Tensor` (`threshold`) and calls `.clamp` on a NumPy array.
  - Python silently up-casts to an *object* array → division by tensor of dtype *object* ⇒ spike rates = noise.
- In `ppo_snn.get_action`:

```python
spk_output, spikes = self.actor(obs_st)
logits = decode_first_spike(spk_output)  # or get_spike_counts …
```

  - `spk_output` is **already averaged**, not a spike train; `decode_first_spike` expects shape `[num_steps, n]` → always returns fallback (~2), so the policy sees constant logits.
- Default `decode_type="lrl"` sidesteps the crash but reduces the model to an ANN with spiking overhead.

**Fix**

1. Make the encoder either full-torch *or* full-numpy; use `torch.clip`.
2. Feed the **spike train** (`spikes`) into decoder functions or treat the read-out directly as logits.

### 2. Very sparse firing

Only **32 time-steps**; with rate code & β = 0.9 you get < 1 spike / neuron.

**Fix**  
`num_steps = 64–128` and scale the pre-normalised observation (e.g. `normalized_obs * 2`).  Target `spike_ratio ≈ 0.25–0.40`.

### 3. Learning-rate & weight initialisation

- LR = **5 e-3** is high for surrogate-gradient SNNs.  
- Manual `fc*.weight += 0.01` biases weights positive → saturation.

**Fix**  
`nn.init.kaiming_uniform_` and remove the manual boost.  LR in **1–3 × 10-4** range; consider cosine decay or warm-up.

### 4. Actor & critic share the same SNN
If either collapses, PPO diverges.  Use an ANN critic until the actor works.

### 5. Reward signal quality

Constant **−1** per step + sparse spikes → mostly negative returns.

**Fix**  
Remove the −1 penalty for now; clip rewards to **[−10, 10]**.

### 6. Categorical logits scaling

Spiking outputs lie in **[0, 1]**; softmax becomes nearly uniform.

**Fix**  
Multiply logits by a temperature (e.g. `logits * 5`) or apply `torch.logit`.

---

## References for quick fixes

- snnTorch RL example – <https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_spike_reinforce.html>
- “SNN4RL” (2022) – Spiking PPO, 128 timesteps, Adam 1e-4, cosine LR.
- “SuperSpike” surrogate gradient – Bellec *et al.*, NeurIPS 2020.

---

## Immediate path (≈ 1 day)

1. Patch encoder/decoder bug; verify `spike_ratio > 0.2`.
2. LR → **1e-4**, `num_steps = 64`, temperature-scale logits.
3. Disable −1 penalty; train 3–5 M steps; monitor *entropy*, *clip_fraction*, *avg_spike_time*.

These three changes alone should let the SNN match the ANN score (in ~2–3× more steps). After basics converge we can explore long-term ideas (event sensors, neuromorphic HW, STDP pre-training, etc.).