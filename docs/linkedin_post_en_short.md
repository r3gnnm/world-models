A month and a half ago I decided to actually understand World Models — so I built one from scratch, solo, on a single GPU, following the JEPA approach Yann LeCun has been championing (his "Path Towards Autonomous Machine Intelligence" paper is what got me hooked).

The model itself worked beautifully: an encoder compresses frames into a latent, a predictor forecasts the next state from an action, trained with VICReg (no pixel reconstruction — predict representations, not pixels). Linear probe R² = 0.999 on agent position.

Then I tried to make it plan — and the agent just stood still.

Diagnosis: Euclidean distance in latent space is a bad planning metric. Correlation with true reachability was only 0.67 — "stand still" scored cheaper than "move toward the goal." I trained a separate temporal-distance metric instead (self-supervised: distance = number of steps between states). Correlation jumped to 0.86, and the agent finally walked through a doorway it couldn't see directly.

The part I'm most proud of came next. LeCun's manifesto describes hierarchical JEPA (H-JEPA) as a concept — fast and slow levels of abstraction — with no concrete recipe for training it. I built a controlled test: compare a trained top-level abstraction against a random, untrained one with identical architecture.

Result: on a fully-observable environment, the trained abstraction was pure decoration — a random projection scored the same (0.995 vs 0.997). Nothing left for it to add. But make it recurrent and remove part of the observation (an egocentric view where room identity isn't recoverable from one frame), and the abstraction becomes measurably real: trained beats random by 24 points (0.638 vs 0.400).

Takeaway: hierarchy only helps when two things hold at once — a mechanism to accumulate history, and information that's genuinely missing without it. This is basically why DreamerV3's RSSM is built the way it is — I arrived at it through experiment, not by reading the diagram.

The real lesson: run the control, not just the metric. Every "surprising" result here had a boring explanation once I tested it against a random baseline. A negative result you can explain beats a positive one you can't.

Open to comparing notes if this space interests you too.

#MachineLearning #WorldModels #JEPA #SelfSupervisedLearning #DeepLearning
