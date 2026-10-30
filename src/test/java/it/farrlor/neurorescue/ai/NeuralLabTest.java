package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class NeuralLabTest {
    @Test
    void closingBeforeQueuedTrainingStartsSkipsModelCreationAndReleasesWorker() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch release = occupy(worker);
        NeuralLab lab = new NeuralLab(worker);
        try {
            CompletableFuture<NeuralLab.TrainingResult> training =
                    lab.train(TrainingScenarios.grounds(), 10, 17, false, ignored -> { });
            lab.close();
            lab.close();
            release.countDown();
            assertThrows(java.util.concurrent.ExecutionException.class, () -> training.get(10, TimeUnit.SECONDS));
            assertTrue(worker.awaitTermination(10, TimeUnit.SECONDS));
            assertFalse(lab.training());
        } finally {
            release.countDown();
            lab.close();
        }
    }

    @Test
    void stoppingQueuedTrainingProducesStoppedOutcomeWithoutAnEpisode() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch release = occupy(worker);
        NeuralLab lab = new NeuralLab(worker);
        try {
            CompletableFuture<NeuralLab.TrainingResult> training =
                    lab.train(TrainingScenarios.grounds(), 10, 17, false, ignored -> { });
            lab.stopTraining();
            release.countDown();
            NeuralLab.TrainingResult result = training.get(10, TimeUnit.SECONDS);
            assertEquals(NeuralLab.TrainingOutcome.STOPPED, result.outcome());
            assertEquals(0, result.completedEpisodes());
        } finally {
            release.countDown();
            lab.close();
            assertTrue(worker.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void closingDuringActiveTrainingCancelsRemainingEpisodes() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        NeuralLab lab = new NeuralLab(worker);
        CountDownLatch progress = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        try {
            var training = lab.train(TrainingScenarios.grounds(), 500, 17, false, ignored -> {
                progress.countDown();
                await(release);
            });
            assertTrue(progress.await(10, TimeUnit.SECONDS));
            lab.close();
            release.countDown();
            NeuralLab.TrainingResult result = training.get(10, TimeUnit.SECONDS);
            assertEquals(NeuralLab.TrainingOutcome.STOPPED, result.outcome());
            assertTrue(result.completedEpisodes() < result.requestedEpisodes());
            assertTrue(worker.awaitTermination(10, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            lab.close();
        }
    }

    @Test
    void closingPausedTrainingWakesWorkerAndCompletesCleanup() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        NeuralLab lab = new NeuralLab(worker);
        CountDownLatch paused = new CountDownLatch(1);
        try {
            var training = lab.train(TrainingScenarios.grounds(), 500, 17, false, ignored -> {
                lab.pause();
                paused.countDown();
            });
            assertTrue(paused.await(10, TimeUnit.SECONDS));
            assertTrue(lab.paused());
            lab.close();
            NeuralLab.TrainingResult result = training.get(10, TimeUnit.SECONDS);
            assertEquals(NeuralLab.TrainingOutcome.STOPPED, result.outcome());
            assertTrue(worker.awaitTermination(10, TimeUnit.SECONDS));
        } finally {
            lab.close();
        }
    }

    @Test
    void completedTrainingShutsDownCleanly() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        NeuralLab lab = new NeuralLab(worker);
        try {
            NeuralLab.TrainingResult result = lab.train(TrainingScenarios.grounds(), 2, 17,
                    false, ignored -> { }).get(10, TimeUnit.SECONDS);
            assertEquals(NeuralLab.TrainingOutcome.COMPLETED, result.outcome());
            assertEquals(result.requestedEpisodes(), result.completedEpisodes());
            lab.close();
            assertTrue(worker.awaitTermination(10, TimeUnit.SECONDS));
        } finally {
            lab.close();
        }
    }

    @Test
    void queuedInferenceDoesNotRunAfterShutdown() throws Exception {
        ExecutorService worker = Executors.newSingleThreadExecutor();
        CountDownLatch release = occupy(worker);
        NeuralLab lab = new NeuralLab(worker);
        var map = TrainingScenarios.grounds().getFirst();
        var observation = new RescueEnvironment(map, RewardScheme.defaults(), 16).observation();
        try {
            var inference = lab.infer(true, map, observation);
            lab.close();
            release.countDown();
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> inference.get(10, TimeUnit.SECONDS));
            assertTrue(failure.getCause().getMessage().contains("closed"));
            assertTrue(worker.awaitTermination(10, TimeUnit.SECONDS));
        } finally {
            release.countDown();
            lab.close();
        }
    }

    private static CountDownLatch occupy(ExecutorService worker) throws InterruptedException {
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        worker.execute(() -> {
            entered.countDown();
            await(release);
        });
        assertTrue(entered.await(10, TimeUnit.SECONDS));
        return release;
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(10, TimeUnit.SECONDS)) throw new AssertionError("Worker gate timed out");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new AssertionError(error);
        }
    }
}
