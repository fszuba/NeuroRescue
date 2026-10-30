package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.EpisodeStatus;
import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Position;
import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import it.farrlor.neurorescue.simulation.Scenario;
import it.farrlor.neurorescue.simulation.StepResult;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

/** Owns the mutable network during a run. Callers must confine all model access to one worker thread. */
public final class TrainingSession {
    private final DqnNetwork network;
    private final DqnConfig config;
    private final ReplayBuffer replay;
    private final ObservationEncoder encoder = new ObservationEncoder();
    private final DqnAgent agent;
    private final Random random;
    private final long seed;
    private final Object pauseLock = new Object();
    private volatile boolean paused;
    private volatile boolean stopRequested;
    private int completedEpisodes;
    private long transitions;

    public TrainingSession(DqnNetwork network, DqnConfig config, long seed) {
        this(network, config, seed, 0);
    }

    public TrainingSession(DqnNetwork network, DqnConfig config, long seed, int completedEpisodes) {
        this.network = network;
        this.config = config;
        this.seed = seed;
        if (completedEpisodes < 0) throw new IllegalArgumentException("Negative completed episode count");
        this.completedEpisodes = completedEpisodes;
        replay = new ReplayBuffer(config.replayCapacity());
        agent = new DqnAgent(network);
        random = new Random(seed);
    }

    public int completedEpisodes() { return completedEpisodes; }
    public long seed() { return seed; }
    public int replaySize() { return replay.size(); }

    public void pause() { paused = true; }
    public void resume() {
        synchronized (pauseLock) { paused = false; pauseLock.notifyAll(); }
    }
    public void stop() { stopRequested = true; resume(); }
    public boolean isPaused() { return paused; }

    public void run(List<Scenario> scenarios, int episodes, Consumer<Progress> listener) {
        runSelected(scenarios, episodes, listener, false);
    }

    public void runCurriculum(int episodes, Consumer<Progress> listener) {
        runSelected(TrainingScenarios.grounds(), episodes, listener, true);
    }

    private void runSelected(List<Scenario> scenarios, int episodes, Consumer<Progress> listener,
                             boolean curriculum) {
        if (scenarios.isEmpty() || episodes < 1) throw new IllegalArgumentException("Training needs maps and episodes");
        for (Scenario scenario : scenarios) encoder.encode(scenario,
                new RescueEnvironment(scenario, RewardScheme.defaults(), config.maxEpisodeSteps()).observation());
        ArrayDeque<EpisodeMetrics> recent = new ArrayDeque<>();
        long started = System.nanoTime();
        int targetEpisodes = completedEpisodes + episodes;
        for (int i = 0; i < episodes && !stopRequested; i++) {
            awaitResume();
            if (stopRequested) break;
            Scenario scenario;
            if (curriculum) {
                double fraction = i / (double) episodes;
                List<Scenario> stage = fraction < 0.5 ? TrainingScenarios.grounds()
                        : fraction < 0.8 ? TrainingScenarios.obstacleCourse() : TrainingScenarios.hazardousCourse();
                scenario = stage.get(completedEpisodes % stage.size());
            } else {
                scenario = scenarios.get(completedEpisodes % scenarios.size());
            }
            float epsilon = config.epsilonAtEpisode(completedEpisodes);
            EpisodeMetrics metrics = trainEpisode(scenario, epsilon);
            if (metrics == null) break;
            completedEpisodes++;
            recent.addLast(metrics);
            if (recent.size() > 20) recent.removeFirst();
            double rollingReward = recent.stream().mapToDouble(EpisodeMetrics::reward).average().orElse(0);
            double rollingSuccess = recent.stream().filter(EpisodeMetrics::success).count() / (double) recent.size();
            double seconds = Math.max(0.001, (System.nanoTime() - started) / 1_000_000_000.0);
            listener.accept(new Progress(completedEpisodes, targetEpisodes, epsilon, metrics,
                    rollingReward, rollingSuccess, replay.size(), transitions / seconds));
        }
    }

    public EpisodeMetrics trainEpisode(Scenario scenario, float epsilon) {
        RescueEnvironment environment = new RescueEnvironment(scenario, RewardScheme.defaults(), config.maxEpisodeSteps());
        double lossSum = 0;
        int updates = 0;
        while (environment.observation().status() == EpisodeStatus.RUNNING) {
            awaitResume();
            if (stopRequested) return null;
            Observation before = environment.observation();
            DqnAgent.Decision decision = agent.choose(scenario, before, epsilon, random);
            Action action = decision.action();
            StepResult step = environment.step(action);
            Observation after = step.observation();
            float learningReward = (float) step.reward() + movementShaping(scenario, before, after, action);
            replay.add(new Experience(encoder.encode(scenario, before), action.ordinal(), learningReward,
                    encoder.encode(scenario, after), ActionMask.valid(scenario, after),
                    after.status() == EpisodeStatus.SUCCESS || after.status() == EpisodeStatus.FAILURE,
                    after.status() == EpisodeStatus.TRUNCATED));
            transitions++;
            if (replay.size() >= config.warmupSteps()) {
                lossSum += network.trainBatch(replay.sample(config.batchSize(), random));
                updates++;
            }
        }
        Observation result = environment.observation();
        return new EpisodeMetrics(result.status(), result.totalReward(), result.steps(),
                result.invalidActions(), result.hazardEncounters(), updates == 0 ? Double.NaN : lossSum / updates);
    }

    // Potential difference for movement only: a back-and-forth route cannot accumulate positive shaping reward.
    static float movementShaping(Scenario scenario, Observation before, Observation after, Action action) {
        if (!action.isMovement() || before.robot().equals(after.robot())) return 0;
        return 0.4f * (goalDistance(scenario, before) - goalDistance(scenario, after));
    }

    private static int goalDistance(Scenario scenario, Observation observation) {
        return (observation.carrying() ? scenario.exits() : observation.victims()).stream()
                .mapToInt(goal -> distance(observation.robot(), goal)).min().orElseThrow();
    }

    private static int distance(Position a, Position b) {
        return Math.abs(a.x() - b.x()) + Math.abs(a.y() - b.y());
    }

    private void awaitResume() {
        synchronized (pauseLock) {
            while (paused && !stopRequested) {
                try { pauseLock.wait(); }
                catch (InterruptedException interrupted) {
                    stopRequested = true;
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    public record Progress(int completedEpisodes, int requestedEpisodes, float epsilon,
                           EpisodeMetrics episode, double rollingReward, double rollingTrainingSuccess,
                           int replaySize, double transitionsPerSecond) { }
}
