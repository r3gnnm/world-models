Last month I posted about building a JEPA world model and finding that hierarchical abstraction only helps under specific conditions. This is the follow-up — about the result I *didn't* publish, and why that mattered more.

I had a promising finding: the more persistent visual landmarks in the environment, the more useful hierarchical memory became. Three data points, clean monotonic trend: 0.6pp → 5.5pp → 8.5pp. It fit the theory. I started drafting it up.

Then I ran it again with three random seeds per point instead of one.

The trend evaporated. Means came back as 5.8, 12.0, 4.0, 4.7, 5.4 — noise, with standard deviation comparable to the differences I'd been excited about. My "law" was three lucky seeds in a row.

What made this worth catching: the qualitative claim underneath survived. Zero landmarks (pure dead reckoning) never produces a real abstraction — that held up across every variant. But the quantitative law didn't exist, and I'd have shipped it.

The same discipline saved me twice more when I moved to a harder testbed — an underactuated 2.5D quadrotor where the drone controls thrust and tilt rather than velocity directly. First, an "action has no effect" reading that turned out to be a metric problem: the ablation compared across different flight trajectories, and in a large world that between-trajectory variance drowned the actual signal. Fixed by comparing within a single trajectory. Second, an apparent fix that looked great on a small smoke test and vanished on the full run.

Each time the pattern was the same: a number that looked like a finding, which stopped looking like one under a control.

Three things I'd take from this into any ML work:

→ A single run is not a result. If you can't afford multiple seeds, you can't afford the claim.
→ A metric that worked at one scale can silently break at another — not because it's buggy, but because its denominator gets dominated by a different noise source.
→ Plot the thing. The clearest diagnosis in the whole project came from one figure of predicted-vs-actual trajectories, after weeks of ambiguous scalar metrics.

Negative results with a diagnosis are worth more than positive results you can't explain. I'd rather publish "here's what I thought I found, and here's what killed it" than a clean curve I can't defend.

#MachineLearning #WorldModels #JEPA #ResearchIntegrity #DeepLearning
