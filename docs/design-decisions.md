# Design decisions and what the experiments showed

Everything under "Evidence" was measured on the **synthetic CICIDS-schema data** used for development (24k rows, heavy-tailed, 9 classes, deliberately hard rare classes). It is directional evidence about the code's behaviour, **not** a result about real traffic. Each point should be re-checked on CICIDS2017.

## Data

**Dedupe before split, fit on train only, cap train only.**
CICIDS2017 contains many exact duplicate flows. If they survive into different splits, the test set contains training examples. The scaler, constant-column filter and NaN medians are fitted on the (capped) train split. Validation and test keep the natural class mix; capping them would change the false-positive denominator and make every alert-volume number meaningless.

**Signed log1p, then min-max.**
Flow features span roughly ten orders of magnitude and include `-1` sentinels (`Init_Win_bytes_*`) and negative IATs. `sign(x) * log1p(|x|)` handles both. Min-max (rather than standardisation) gives each feature a bounded range, which the sigmoid-output GAN and the spike encoder need. Test and serve-time values are **not clipped** by the preprocessor, because for the VAE an out-of-range value is the anomaly signal; the GAN and SNN clip internally where they must. An infinite value at serve time is mapped to a huge finite one, so it reads as extremely anomalous instead of crashing or being silently clipped.

**Label grouping.** Default `family` grouping (9 classes) merges the DoS variants, the two Patator brute-forces and the three web attacks. Rare per-variant classes (SQL injection has 21 flows) cannot be split three ways. `label_grouping: raw` keeps all 15.

## VAE

**KL weight must be small.** With min-max-scaled flow features the total variance is low, so a conventional `beta = 0.5` collapsed the posterior (KL about 0.007, zero active latent dimensions, AUROC 0.75). `beta = 0.01` kept all 16 latent dimensions active. The trainer logs `active_latent_dims` as a collapse diagnostic.

**Score = reconstruction error + latent Mahalanobis distance**, each log-transformed and robust-standardised on benign *validation* flows, then mapped through the benign-validation empirical CDF. Score 0 means "typical benign", 1 means "beyond almost every benign validation flow", so the number keeps its meaning across retrains. The floor (0.95) and ceiling (0.999) quantiles are configurable.

## GAN

**Why WGAN-GP, conditional.** Rare classes are exactly the small-sample regime where vanilla GANs oscillate or mode-collapse. The Wasserstein critic gives a gradient even when real and fake barely overlap; the gradient penalty avoids weight clipping. One conditional model shares structure across attack families.

**Guard-rails.** Minimum 20 real samples per class; at most 20 synthetic per real; train split only; output snapped to real point masses.

**Evidence on snapping.** Real flow data is full of exact zeros, one/zero flags and sentinels. A sigmoid generator emits 0.0003 where real data has 0, which a classifier detects instantly (and then learns as a shortcut). After snapping onto the real atoms, real-vs-synthetic discriminator AUC moved only from 1.00 to 0.97-1.00 depending on the class (synthetic data, 2,500 iterations): marginal moments improved, but the samples are **still easy to tell apart**. That is why the fidelity report exists and why the ablation must decide whether augmentation actually helps; do not assume it does.

## SNN

**Direct vs rate encoding (development evidence, synthetic data, same network and seed).**

| Encoding | Macro-F1 | Hidden firing rates |
|---|---|---|
| rate (Bernoulli, T=20) | 0.65 | 0.21 / 0.31 |
| direct | 0.87 | 0.14 / 0.27 |

Bernoulli sampling noise over 20 steps swamps the small differences between continuous features. Direct encoding feeds the analog value as input current to the first layer, which converts it to spikes from then on. It is deterministic and standard in current SNN work. `rate` stays available for hardware that only accepts spike-coded input.

**Readout.** The output layer integrates current without spiking or resetting; logits are its time-averaged membrane potential. This avoids ties between output spike counts and maps to an accumulator in hardware.

**Sparsity regulariser.** Penalising mean firing rate directly from step 0 killed the network entirely on a small configuration (hidden layer rate 0.000, macro-F1 0.09). The final form is a hinge above `rate_target` (only excess firing is penalised) with a linear warm-up. Evidence (256/128 hidden, T=20, synthetic data):

| rate_reg | Macro-F1 | Hidden rates | Est. energy vs ANN |
|---|---|---|---|
| 0 | 0.872 | 0.14 / 0.27 | 0.71 |
| 1 | 0.885 | 0.047 / 0.053 | 0.46 |
| 3 | 0.893 | 0.045 / 0.042 | 0.45 |

The floor of about 0.45 is dominated by the first layer's analog multiply-accumulates under direct encoding. Under `rate` encoding the input spikes dominate instead, and in development the SNN needed more operations than the ANN (energy ratio about 1.1), so the "low-power" claim is only true with sparse activity.

**Energy proxy definition.** Accumulates (0.9 pJ) for every spike times its fan-out, MACs (4.6 pJ) for analog first-layer inputs, versus MACs for a same-shape dense ANN (45 nm, 32-bit float, Horowitz 2014). It ignores neuron updates and memory movement. It is for comparing configurations, not a hardware measurement.

**Temperature scaling.** Fitted on the full natural-prevalence validation logits so `1 - P(Benign)` behaves like a probability inside the risk formula.

## Fusion

Noisy-OR so either detector can raise the alarm; `anomaly_weight = 0.8` keeps the noisier unsupervised signal from reaching risk = 1 by itself. Thresholds are quantiles of *benign validation* risk at the target false-positive rates, with floors (`min_*_threshold`) so a degenerate benign distribution cannot push a threshold to zero. On the synthetic data the fused risk reached AUROC 0.99 where the VAE alone reached 0.72, which is the intended complementarity; this needs confirming on real data.

## Things deliberately not done yet

- No hyper-parameter search; defaults come from a handful of development runs.
- No cross-dataset evaluation, no drift monitoring, no authentication on the API.
- No real neuromorphic hardware run; latency is for the PyTorch simulation.
