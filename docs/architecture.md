# Architecture

NeuroRescue remains one Maven module. The domain and training code have no JavaFX dependency.

| Package | Responsibility | Dependencies |
| --- | --- | --- |
| `simulation` | Validated maps, deterministic transitions, rewards, immutable observations | Java standard library |
| `baseline` | Complete-map shortest safe route | `simulation` |
| `ai` | Encoding, DQN, replay, training, evaluation, persistence | `simulation`, `baseline`, DJL |
| root package | JavaFX coordination, simulation rendering, training controls, and charts | all domain packages, JavaFX |

`RescueEnvironment` is confined to one thread and returns immutable `Observation` and `StepResult` records. `Scenario.fromAscii` checks dimensions, entities, and safe reachability. `Scenarios.generated(seed)` retries until a safe route exists. `reset(long)` has a seed parameter for possible future stochastic mechanics; current episodes are deterministic after map creation.

`NeuralLab` serializes DJL operations on one worker. A training request has a cancellation flag before its `TrainingSession` exists, so stopping or closing the app can cancel queued as well as active work. Closing is idempotent: queued operations check the closed state, active training receives a stop request, and model cleanup is queued after the worker's current task. The JavaFX thread does not wait for the worker to terminate. A paused session is woken for cancellation.

Training publishes immutable progress records, with greedy evaluation every 20 episodes and UI delivery throttled to approximately five updates per second. `TrainingPanel` presents the controls, status, metrics, and charts; `TrainingChartData` retains bounded samples with their real episode indices. Missing warmup loss is omitted, and evaluation samples are drawn only at measured episodes. `SimulationView` owns map rendering and movement animation, while `MissionPanel` presents the live episode inspector. `NeuroRescueApplication` coordinates modes and callbacks; JavaFX applies worker results through `Platform.runLater`. Neural simulation inference is queued on the same worker. A generation counter discards stale inference results after reset, agent change, or mode change.

The DQN file is a versioned binary containing metadata and finite float parameters. It is written to a temporary file in the destination directory, then moved into place atomically where supported. Loading validates the format, encoder version, action order, shape, and parameter names before replacing the active model. The file stores inference weights and provenance only; replay and optimizer state are not restored.

See [learning design](reinforcement-learning.md), [experiment](experiments.md), [foundation decision](decisions/0001-foundation.md), and [Maven migration](decisions/0002-maven-migration.md).
