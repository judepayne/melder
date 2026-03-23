  1. Lower learning rate (2e-5 → 1e-5 → 5e-6) — smaller weight updates per step, less disruption to pre-trained
  knowledge
  2. LoRA instead of full fine-tune — only update ~1% of parameters via low-rank adapters, the rest stay frozen.
  This is probably the strongest anti-forgetting measure
  3. MNRL temperature (scale 20.0 → 10.0) — softens the contrastive signal, less aggressive separation
  4. Weight decay (0.01 → 0.05) — penalises large weight changes, keeps parameters closer to pre-trained values
  5. Freeze lower layers — only fine-tune the top N transformer layers, preserving low-level language
  understanding
