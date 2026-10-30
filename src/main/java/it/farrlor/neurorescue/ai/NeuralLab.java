package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Scenario;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

/** Serializes every DJL operation on one worker; pause and stop only modify session flags. */
public final class NeuralLab implements AutoCloseable {
    private final Object lifecycle = new Object();
    private final ExecutorService worker;
    private volatile TrainingRun activeRun;
    private volatile boolean closed;
    private DqnNetwork model;
    private DqnNetwork untrainedReference;
    private long modelSeed;
    private int completedEpisodes;

    public NeuralLab() {
        this(Executors.newSingleThreadExecutor(r -> new Thread(r, "neurorescue-ml")));
    }

    NeuralLab(ExecutorService worker) {
        this.worker = Objects.requireNonNull(worker);
    }

    public boolean training() {
        return activeRun != null;
    }
    public boolean paused() {
        TrainingRun run = activeRun;
        return run != null && run.paused();
    }
    public void pause() {
        TrainingRun run = activeRun;
        if (run != null) run.pause();
    }
    public void resume() {
        TrainingRun run = activeRun;
        if (run != null) run.resume();
    }
    public void stopTraining() {
        TrainingRun run = activeRun;
        if (run != null) run.stop();
    }

    private <T> CompletableFuture<T> submit(Callable<T> operation) {
        synchronized (lifecycle) {
            if (closed) return CompletableFuture.failedFuture(new IllegalStateException("Neural lab is closed"));
            CompletableFuture<T> result = new CompletableFuture<>();
            worker.execute(() -> {
                if (closed) {
                    result.completeExceptionally(new IllegalStateException("Neural lab is closed"));
                    return;
                }
                try {
                    result.complete(operation.call());
                } catch (Throwable error) {
                    result.completeExceptionally(error);
                }
            });
            return result;
        }
    }

    public CompletableFuture<Void> newModel(long seed) {
        return submit(() -> {
            DqnNetwork replacement = new DqnNetwork(DqnConfig.defaults(), seed);
            if (model != null) model.close();
            model = replacement;
            modelSeed = seed;
            completedEpisodes = 0;
            return null;
        });
    }

    public CompletableFuture<ModelState> load(Path file) {
        return submit(() -> {
            DqnNetwork.LoadedModel loaded = DqnNetwork.load(file);
            if (model != null) model.close();
            model = loaded.network();
            modelSeed = loaded.seed();
            completedEpisodes = loaded.completedEpisodes();
            return new ModelState(modelSeed, completedEpisodes);
        });
    }

    public CompletableFuture<Void> save(Path file) {
        return submit(() -> {
            if (model == null) throw new IllegalStateException("Create or load a model first");
            model.save(file, modelSeed, completedEpisodes);
            return null;
        });
    }

    public CompletableFuture<DqnAgent.Decision> infer(boolean untrained, Scenario scenario, Observation observation) {
        return submit(() -> {
            DqnNetwork selected;
            if (untrained) {
                if (untrainedReference == null) untrainedReference = new DqnNetwork(DqnConfig.defaults(), 0);
                selected = untrainedReference;
            } else {
                if (model == null || completedEpisodes == 0) throw new IllegalStateException("No trained model is available");
                selected = model;
            }
            return new DqnAgent(selected).choose(scenario, observation, 0, new Random(0));
        });
    }

    public enum EvaluationAgent { ROUTE, RANDOM, UNTRAINED, TRAINED }

    public CompletableFuture<Evaluation.Summary> evaluate(EvaluationAgent agent, List<Scenario> scenarios,
                                                           int episodes, long seed, int maxSteps) {
        return submit(() -> switch (agent) {
            case ROUTE -> Evaluation.baseline(scenarios, episodes, seed, maxSteps);
            case RANDOM -> Evaluation.random(scenarios, episodes, seed, maxSteps);
            case UNTRAINED -> {
                if (untrainedReference == null) untrainedReference = new DqnNetwork(DqnConfig.defaults(), 0);
                yield Evaluation.neural(untrainedReference, scenarios, episodes, seed, maxSteps);
            }
            case TRAINED -> {
                if (model == null || completedEpisodes == 0) throw new IllegalStateException("No trained model is available");
                yield Evaluation.neural(model, scenarios, episodes, seed, maxSteps);
            }
        });
    }

    public CompletableFuture<TrainingResult> train(List<Scenario> scenarios, int episodes, long seed,
                                                    boolean curriculum, Consumer<TrainingSnapshot> listener) {
        synchronized (lifecycle) {
            if (closed) return CompletableFuture.failedFuture(new IllegalStateException("Neural lab is closed"));
            if (activeRun != null) throw new IllegalStateException("Training is already running");
            TrainingRun run = new TrainingRun();
            activeRun = run;
            CompletableFuture<TrainingResult> result = new CompletableFuture<>();
            worker.execute(() -> {
                TrainingResult completed = null;
                Throwable failure = null;
                try {
                    if (closed) throw new IllegalStateException("Neural lab closed before training started");
                    completed = runTraining(run, scenarios, episodes, seed, curriculum, listener);
                } catch (Throwable error) {
                    failure = error;
                } finally {
                    activeRun = null;
                }
                if (failure == null) result.complete(completed);
                else result.completeExceptionally(failure);
            });
            return result;
        }
    }

    private TrainingResult runTraining(TrainingRun run, List<Scenario> scenarios, int episodes, long seed,
                                       boolean curriculum, Consumer<TrainingSnapshot> listener) {
        int targetEpisodes = completedEpisodes + episodes;
        if (run.stopped()) return new TrainingResult(completedEpisodes, targetEpisodes, TrainingOutcome.STOPPED);
        if (model == null) {
            model = new DqnNetwork(DqnConfig.defaults(), seed);
            completedEpisodes = 0;
            targetEpisodes = episodes;
        }
        modelSeed = seed;
        TrainingSession session = new TrainingSession(model, DqnConfig.defaults(), seed, completedEpisodes);
        run.bind(session);
        long[] lastPublish = {0};
        int[] lastPublishedEpisode = {completedEpisodes};
        TrainingSession.Progress[] latestProgress = {null};
        Consumer<TrainingSession.Progress> progressListener = progress -> {
            completedEpisodes = progress.completedEpisodes();
            latestProgress[0] = progress;
            long now = System.nanoTime();
            boolean evaluate = completedEpisodes % 20 == 0;
            if (evaluate || now - lastPublish[0] >= 200_000_000L
                    || progress.completedEpisodes() == progress.requestedEpisodes()) {
                Evaluation.Summary evaluation = evaluate && !run.stopped() && !closed
                        ? Evaluation.neural(model, scenarios, 12, 2026, DqnConfig.defaults().maxEpisodeSteps())
                        : null;
                listener.accept(new TrainingSnapshot(progress, evaluation));
                lastPublish[0] = now;
                lastPublishedEpisode[0] = progress.completedEpisodes();
            }
        };
        if (curriculum) session.runCurriculum(episodes, progressListener);
        else session.run(scenarios, episodes, progressListener);
        if (!closed && latestProgress[0] != null
                && lastPublishedEpisode[0] != latestProgress[0].completedEpisodes()) {
            listener.accept(new TrainingSnapshot(latestProgress[0], null));
        }
        TrainingOutcome outcome = completedEpisodes == targetEpisodes
                ? TrainingOutcome.COMPLETED : TrainingOutcome.STOPPED;
        return new TrainingResult(completedEpisodes, targetEpisodes, outcome);
    }

    @Override
    public void close() {
        synchronized (lifecycle) {
            if (closed) return;
            closed = true;
            stopTraining();
            worker.execute(() -> {
                try {
                    if (model != null) model.close();
                } finally {
                    if (untrainedReference != null) untrainedReference.close();
                }
            });
            worker.shutdown();
        }
    }

    private static final class TrainingRun {
        private TrainingSession session;
        private boolean paused;
        private boolean stopped;

        synchronized void bind(TrainingSession session) {
            this.session = session;
            if (stopped) session.stop();
            else if (paused) session.pause();
        }

        synchronized void pause() {
            paused = true;
            if (session != null) session.pause();
        }

        synchronized void resume() {
            paused = false;
            if (session != null) session.resume();
        }

        synchronized void stop() {
            stopped = true;
            paused = false;
            if (session != null) session.stop();
        }

        synchronized boolean paused() {
            return paused;
        }

        synchronized boolean stopped() {
            return stopped;
        }
    }

    public record ModelState(long seed, int completedEpisodes) { }
    public enum TrainingOutcome { COMPLETED, STOPPED }
    public record TrainingResult(int completedEpisodes, int requestedEpisodes, TrainingOutcome outcome) { }
    public record TrainingSnapshot(TrainingSession.Progress progress, Evaluation.Summary evaluation) { }
}
