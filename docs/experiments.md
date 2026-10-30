# Controlled learning experiment

Run the longer, opt-in JUnit experiment with `./mvnw '-Dtest=LearningExperimentTest' '-Dneurorescue.experiment=true' test` (use `.\mvnw.cmd` on Windows). It initializes a fresh DQN with seed 17, evaluates without exploration, trains for 360 episodes across three 5×5 grounds, evaluates every 60 episodes, saves an inference model, reloads it, and compares Q values within `1e-5`.

The Windows x86-64 CPU run was repeated on 30 October 2026 after the lifecycle and JavaFX refactor. It reported:

| Policy / point | Training-ground rescues | Mean raw reward | Mean steps |
| --- | ---: | ---: | ---: |
| Fresh DQN | 0 / 30 | -3.20 | 32.0 |
| Random valid actions | 5 / 30 | 4.32 | 29.2 |
| Shortest safe route | 30 / 30 | 36.57 | 4.33 |
| DQN after 360 episodes | 30 / 30 | 36.57 | 4.33 |
| Trained DQN on held-out layouts | 0 / 20 | -2.20 | 32.0 |

At episode 60, greedy evaluation on the training grounds already reached 30/30. The final episode's measured mean optimization loss was approximately 0.222. Loss is not an outcome metric; the rescue rate is the evidence of learning on this controlled task. The held-out result shows that this network has not learned robust transfer to new victim and exit placements.

These maps and episodes are deterministic. Thirty episodes rotate through three layouts, so the 30 outcomes are repeated evaluations of three distinct cases, not 30 independent samples. The environment's `reset(seed)` currently does not randomize within an episode. UI evaluation uses a fixed seed and rotates through the selected map set, reporting success rate, mean environment reward, mean length, invalid actions, and hazards. Generalization claims require more varied held-out maps and training seeds in future work.
