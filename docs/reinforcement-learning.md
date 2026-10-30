# DQN design and rewards

The baseline is a complete-map shortest safe route policy. The neural agent uses a real local DQN trained with DJL 0.38.0 and CPU PyTorch 2.7.1. They observe the same map, although the DQN is not guaranteed to generalize across layouts.

## Environment rewards

| Event | Reward |
| --- | ---: |
| Every action | -0.1 |
| Invalid action | additional -1.0 |
| Pickup | +2.0 |
| Delivery | +15.0 |
| Final delivery | additional +20.0 |
| Entering a hazard | -10.0 and episode failure |

Pickup and delivery bonuses are bounded by the number of victims. Invalid actions consume a step. A finite episode limit prevents unlimited wandering. The trainer adds `0.4 × (old Manhattan distance − new Manhattan distance)` for valid movement toward the current victim or exit. Moving back reverses that shaping reward, so a two-tile loop cannot profit from it. The target changes only at pickup or delivery, which are rewarded by the environment. Evaluation and displayed episode rewards use the raw environment totals.

## State, actions, and updates

The encoder uses six 12×12 binary channels: walls, obstacles, hazards, exits, remaining victims, and robot. Outside-map cells are walls. Seven features record carrying state, normalized robot coordinates, and relative offsets to the nearest victim and exit. Ties use row then column order for deterministic encoding. Maps larger than 12×12 are rejected for neural use rather than silently resized.

The network is a 871→64→64→6 multilayer perceptron with ReLU hidden layers. Output order is UP, DOWN, LEFT, RIGHT, PICK_UP, DROP_OFF. The default trainer uses learning rate 0.001, discount 0.95, epsilon from 1.0 down to 0.05 with factor 0.995 per completed episode, replay capacity 10,000, batch size 32, warmup 128 transitions, target synchronization every 200 updates, and 64 steps per training episode. Parameters are configurable in code through `DqnConfig`; the UI exposes episode count, seed, and map curriculum to avoid excessive controls.

Both exploration and greedy action choice mask impossible moves, premature pickup, and premature drop-off. Hazard tiles are traversable and stay available, so avoiding them must be learned. The same mask applies when selecting the maximum target Q value. The update target is `r` for success or failure, and `r + γ max_valid Q_target(s′,a′)` for ongoing or truncated transitions. A truncation ends the sampled episode but still bootstraps, whereas a terminal transition does not. Huber loss, Adam, replay minibatches, and a frozen periodically copied target network provide the actual gradient-based updates.

The **Progressive curriculum** option spends the first 50% of requested episodes on basic rescue grounds, the next 30% on obstacle courses, and the final 20% on hazard courses. Each stage uses multiple fixed, validated layouts. It is a training schedule, not a claim that the later stages have converged. The simpler controlled experiment trains only on the three grounds; its held-out evaluation is reported separately.

The JavaFX dashboard receives progress at throttled intervals. Chart points retain their actual episode numbers; the reward chart shows raw episode reward alongside the 20-episode mean computed by the trainer. Loss is absent during replay warmup, and greedy evaluation success appears only at measured evaluation episodes. The map displayed during training is a scenario preview, not a replay of the current episode.
