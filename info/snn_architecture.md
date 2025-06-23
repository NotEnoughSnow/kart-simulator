Network quick-facts (today’s version)
• Two fully-connected LIF layers (64 neurons each)
• Shared weights for actor/critic
• No recurrence, no skip connections, single read-out linear layer
• Fixed β=0.9 leak, identical time-constants across neurons
• Only rate/temporal-average decoding

That is essentially the SNN analogue of a 2-layer MLP (“basic” but a good baseline). Capacity vs. convergence in spiking RL is not the same as with ANNs; scaling must keep spike statistics healthy. Here is a concise playbook:

First get the current width working
You’ll know it works when
• spike_ratio ≈ 0.25–0.40
• entropy doesn’t collapse early
• returns plateau at a sensible value. Only then start changing size.
If the policy under-fits (score plateaus well below ANN)
a. Width: 64 → 128 (both hidden layers).
Keep LR, β, num_steps identical; watch spike_ratio.
b. Depth: add a third LIF layer of 64 units.
Bigger and deeper will usually need more timesteps (e.g. 96–128). Signs of too much capacity:
• entropy plunges to 0 quickly;
• spike_ratio shoots to >0.6;
• weight-change metrics explode despite grad-clipping.
If the policy over-fits / is unstable
a. Narrow to 32 units per layer.
b. Add dropout on the read-out rate (nn.Dropout(p=0.3) after avg_spk2).
c. Increase weight decay (e.g. Adam(..., weight_decay=1e-4)).
Architectural upgrades that usually help SNN-RL
• Separate actor / critic networks (different initialisation + dropout).
• Heterogeneous neuron parameters: draw β ∈ U(0.8, 0.95); this reduces synchrony.
• Spike-frequency adaptation (SNN-Torch: snn.AdaptiveLIF).
• Recurrent head: feed last layer spikes back to its input (small alpha weight).
• “Residual” current: x → fc → LIF + skip connection of the raw current.
Encoding alternatives (gains bigger than making the net wider)
Rate + 32 steps is wasteful. Try:
• Latency code (first-spike) with 64 steps – network can stay the same size.
• Population code: project obs through random Gaussian matrix → higher-dim spike inputs, smaller hidden size.
Practical hyper-parameter grid to run (ascending complexity)
hidden	depth	num_steps	notes
64	2	64	Baseline after bug-fixes
128	2	64	More width
64	3	96	More depth
32	2	64	Smaller, faster – good for ablation
64	2	64	β heterogeneous, SFA
Use the same total training steps for fairness. Track: return, entropy, spike_ratio.
Monitoring hints
• Keep avg_spike_time roughly mid-range (not always 1 or num_steps).
• If clip_fraction in PPO stays >0.3 after a few epochs → network too large / LR high.
• Visualise weight histograms; spiking nets often drift towards all-positive weights – apply weight decay or re-centre every few epochs.
Bottom line
Fix the encoder/decoder and firing-rate issues first; you might already reach reasonable returns with the 2×64 SNN. Afterwards, scale width (×2) before depth, add heterogeneity, and only increase num_steps when the spike ratio becomes too low.