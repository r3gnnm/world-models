A world model that encodes position perfectly can still be useless for planning. Five experiments on why.

I tested whether a JEPA-style world model can fly a quadrotor to a goal specified as an image — no external localization, just pixels. The drone is underactuated: it controls thrust and tilt, and horizontal motion exists *only* as a consequence of tilting.

First I established a control: same planner, but running on the *true physics* instead of the learned model. 10/10 goals reached. That one number saved me weeks — every later failure was now unambiguously the model's fault, not the planner's.

Then:

**Egocentric view**: tilt encoded beautifully (probe R²=0.92), position poorly (0.53). Planning 0/10.

**Global view, random maps per episode**: everything collapsed to negative R², action-gap flat at zero — *while training loss went down*. Diagnosis: between-episode variance (map identity) was 5.1× the within-episode variance (drone motion), and maps are static within an episode. Cheapest way to cut prediction loss? Encode the background, ignore the agent. Textbook shortcut learning — invisible if you only watch the loss.

**Global view, one fixed map**: shortcut closed, position near-perfect (0.99). But tilt fell to 0.34 — the drone is a small mark in a downsampled frame. Planning 1/10.

**Both views, two channels**: position AND tilt excellent (0.98 / 0.94). Planning 2/10 — still failing, because velocity was missing (0.28), and an inertial system has to know when to brake.

**Two channels + recurrent context**: velocity 0.28 → 0.77. Planning 5/10, mean distance-to-goal 20.6 → 15.0.

Monotonic. Every closed information deficit improved control.

The velocity result has my favorite control of the project: trained GRU vs *untrained* GRU vs a random **instantaneous** linear projection of matched dimensionality. The linear projection gave exactly 0.000 gain on every variable — mathematically expected, since a linear map of the latent adds nothing a linear probe couldn't already extract. That zero is what proves the gain came from recurrence, not from "more features."

Three takeaways:

→ Shortcut learning doesn't show up in the loss. It showed up in the *contradiction* between loss (improving) and probe (collapsing).
→ Observation designs can be complementary. The "better" view was worse; the combination beat both.
→ What's structurally absent from a single frame stays absent no matter how long you train.

Engineering conclusion: a purely visual world model must recover position, attitude *and* velocity from pixels before latent goal-image planning works. In a real drone those come from VIO/SLAM and an IMU — which is why the practical architecture is hybrid: classical state estimation underneath, world model on top for prediction and uncertainty. I didn't assume that. I measured my way into it.

#MachineLearning #WorldModels #JEPA #Robotics #DeepLearning
