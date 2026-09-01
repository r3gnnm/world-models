A month and a half ago I decided to actually understand how World Models work — so instead of just reading papers, I built one from scratch. Solo, on a single GPU, following the JEPA approach that Yann LeCun has been championing.

I first got interested in this after reading his "Path Towards Autonomous Machine Intelligence" paper. What I didn't expect was that the most valuable part of this project wouldn't be where things worked — it would be where they broke.

🧩 What I built
An encoder compresses observations into a latent representation; a predictor learns to forecast the next state from an action — not in pixel space, but directly in the latent (VICReg regularization to prevent representation collapse). Result: linear probe R² = 0.999 — the agent's position is almost perfectly recoverable from the latent alone.

The model was excellent. Then I tried to make it plan — and the agent just... stood still.

🔍 First diagnosis
Turns out Euclidean distance in latent space is a bad planning metric. Correlation with actual reachability: only 0.67. The planner kept finding "cheap" local minima where staying put looked better than moving.
Fix: train a separate temporal-distance metric (distance = number of steps between states, self-supervised from the same trajectories). Correlation jumped to 0.86 — and the agent finally walked through a doorway it couldn't see directly.

🧪 Stress tests
Next, I tried to break it on purpose:
→ More complex topology (three rooms instead of two): no degradation.
→ An independent "noise" object moving in the scene: the model partially ignores it, as JEPA theory predicts — but not fully, the action signal gets diluted threefold.
→ Partial observability (egocentric view) breaks the memoryless architecture entirely. Added a GRU, and ran a controlled experiment showing the gain comes specifically from memory, not just more parameters — the same GRU gives zero benefit on the fully-observable environment.

🏗️ Hierarchy — and an unexpected result
Then I tried something that's described in the literature only as a concept (H-JEPA, from LeCun's 2022 manifesto) — a hierarchical model with a "fast" and a "slow" level of abstraction. I built a controlled test: compare a trained top level against a random, untrained one.

Result: on a fully-observable environment, the top level was purely decorative — a random projection scored the same as the trained one (0.995 vs 0.997). There was nothing left for the abstraction to add; everything was already available below.
But make the top level recurrent and remove part of the observation, and the abstraction becomes measurably real — the trained version beats the random one by 24 percentage points.

Takeaway: hierarchy only helps when two conditions hold at once — there's a mechanism to accumulate history, AND the information genuinely isn't available below without it. Neither condition alone guarantees anything. This is basically the architectural logic behind DreamerV3's RSSM — except I arrived at it through experiment, not by copying the paper.

💡 The real lesson
Not blind iteration, but diagnosis at every failure: why it didn't work, what exactly broke, how to prove it with a controlled experiment rather than a guess. That's what turns a toy project into a small but real piece of research.

World Models are having a moment right now — DeepMind's Genie 3, Fei-Fei Li's World Labs (Marble), LeCun's new venture (AMI Labs, $1B+ seed). Glad I got to engage with this field not as someone running other people's code, but as someone who found and fixed what broke, one experiment at a time.

Open to conversations if this space interests you too — as a researcher or as a practitioner.

#MachineLearning #WorldModels #JEPA #SelfSupervisedLearning #DeepLearning #AIResearch
