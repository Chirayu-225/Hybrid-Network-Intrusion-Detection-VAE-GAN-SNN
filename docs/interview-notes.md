# Interview notes

A one-page cheat sheet for talking about this project. Fill the numbers from your own `reports/snn/summary.md`; do not quote anything you have not generated.

## 30-second pitch

"I built an intrusion detection pipeline for the two things that make real deployments hard: analysts get buried in false positives, and rare attacks have too few labels. A VAE trained only on benign traffic scores how abnormal a flow is, so it can catch attacks it has never seen. A conditional WGAN-GP augments the rare attack classes in the training set only. A spiking neural network does the classification with sparse, event-driven activity, which is what you want on a router or IoT gateway. A fusion layer combines the two signals into a risk score and an allow, monitor or block action, with thresholds calibrated to a target false-positive rate. It runs behind a FastAPI service with a Streamlit dashboard, in Docker."

## Questions to expect, and honest answers

**Why three models instead of one good classifier?**
They cover different failure modes. A supervised model only knows the attack families in its labels; the VAE only knows what normal looks like, so it is the zero-day signal; the GAN helps the supervised model on classes with few examples. Fusing them means either detector can raise the alarm.

**How do you know your numbers are not inflated?**
Duplicates are removed before splitting, scaler and filters are fitted on train only, synthetic data never touches validation or test, and thresholds are calibrated on validation and only reported on test. The test set keeps the natural class prevalence, so false-positive and alert-volume numbers are realistic. I also drop IPs, timestamps and destination port so models cannot memorise hosts.

**Did the GAN actually help?**
I do not assume it does. The fidelity report shows whether synthetic flows are distinguishable from real ones (on my development data they still were), and the ablation (baseline vs +GAN per-class F1) is the test. If it does not help on a class, I would say so.

**Why is the SNN better than a small MLP?**
Not on accuracy: a same-shape ANN should match or beat it. The case is operations per inference on event-driven hardware. My energy figure is a proxy (synaptic ops with 45 nm per-op energies), the simulation runs on a CPU/GPU, and with dense or rate-coded activity the SNN actually needs *more* operations. The benefit only appears when firing is sparse, which is why I added a firing-rate penalty and report firing rates.

**Why direct encoding and not rate coding?**
In my development runs rate coding lost a lot of accuracy because Bernoulli sampling noise swamps small differences between continuous features. Direct encoding feeds the analog value into the first layer, which spikes from there on. Rate coding is still a config switch if the target hardware needs spike-coded input.

**What happened with posterior collapse?**
With a standard KL weight the VAE collapsed (zero active latent dimensions). Min-max scaled flow features have low total variance, so the KL term dominated. A much smaller beta fixed it, and the trainer now logs active latent dimensions as a diagnostic.

**How do you pick the action thresholds?**
As quantiles of benign validation risk at configured false-positive targets (0.5% at block, 3% at monitor), with floors so a degenerate distribution cannot set a threshold to zero. That turns "what threshold?" into a business question: how many benign flows can the SOC tolerate reviewing.

**What would you do next?**
The ablation study (baseline vs +GAN, unknown-attack hold-out, SNN vs ANN), cross-dataset evaluation on UNSW-NB15, and drift monitoring on the API.

**What are the limitations?**
One 2017 lab dataset with known labelling issues; Heartbleed and Infiltration have too few flows to judge; energy numbers are a proxy; the API has no authentication; low-and-slow attacks that mimic benign flows will evade the VAE.

## Context worth mentioning

- Alert volume is the practical metric a SOC cares about; the funnel table in `summary.md` shows it directly.
- If you want an India-specific angle, security teams there work under CERT-In incident-reporting obligations; check the current wording of those directions before you quote them.
