package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class TrainingSessionTest {
    private static DqnConfig smallConfig() {
        return new DqnConfig(0.001f, 0.95f, 1, 0.05f, 0.99f, 64, 8, 8, 5, 16, 16);
    }

    @Test
    void pauseResumeAndStopKeepEpisodeCountsHonest() throws Exception {
        DqnConfig config = smallConfig();
        try (DqnNetwork network = new DqnNetwork(config, 7)) {
            TrainingSession session = new TrainingSession(network, config, 7);
            session.pause();
            CountDownLatch started = new CountDownLatch(1);
            CompletableFuture<Void> run = CompletableFuture.runAsync(() -> {
                started.countDown();
                session.run(TrainingScenarios.grounds(), 4, ignored -> { });
            });
            assertTrue(started.await(10, TimeUnit.SECONDS));
            assertEquals(0, session.completedEpisodes());
            assertFalse(run.isDone());
            session.resume();
            run.get(10, TimeUnit.SECONDS);
            assertEquals(4, session.completedEpisodes());
            assertTrue(session.replaySize() > 0);
            assertTrue(session.replaySize() <= config.replayCapacity());
            assertTrue(network.optimizationSteps() > 0);

            TrainingSession stopped = new TrainingSession(network, config, 8);
            stopped.pause();
            CompletableFuture<Void> cancelled = CompletableFuture.runAsync(() ->
                    stopped.run(TrainingScenarios.grounds(), 50, ignored -> { }));
            stopped.stop();
            cancelled.get(10, TimeUnit.SECONDS);
            assertEquals(0, stopped.completedEpisodes());
        }
    }

    @Test
    void rollingRewardUsesActualEpisodesRatherThanPublishedSnapshotSpacing() {
        DqnConfig config = smallConfig();
        try (DqnNetwork network = new DqnNetwork(config, 17)) {
            TrainingSession session = new TrainingSession(network, config, 17);
            java.util.ArrayDeque<Double> recent = new java.util.ArrayDeque<>();
            session.run(TrainingScenarios.grounds(), 5, progress -> {
                recent.addLast(progress.episode().reward());
                assertEquals(recent.stream().mapToDouble(Double::doubleValue).average().orElseThrow(),
                        progress.rollingReward(), 1e-9);
            });
        }
    }

    @Test
    void evaluationUsesRealEpisodesAndDoesNotTrain() {
        var maps = TrainingScenarios.grounds();
        var baseline = Evaluation.baseline(maps, 9, 23, 32);
        assertEquals(9, baseline.successes());
        assertEquals(0, baseline.averageInvalidActions());
        assertTrue(baseline.averageReward() > 30);
        try (DqnNetwork network = new DqnNetwork(smallConfig(), 7)) {
            int before = network.optimizationSteps();
            Evaluation.neural(network, maps, 9, 23, 32);
            assertEquals(before, network.optimizationSteps());
        }
    }

    @Test
    void actionMaskAndExplorationNeverSelectImpossibleActions() {
        var scenario = TrainingScenarios.grounds().getFirst();
        var observation = new RescueEnvironment(scenario, RewardScheme.defaults(), 16).observation();
        boolean[] valid = ActionMask.valid(scenario, observation);
        assertFalse(valid[Action.UP.ordinal()]);
        assertFalse(valid[Action.PICK_UP.ordinal()]);
        try (DqnNetwork network = new DqnNetwork(smallConfig(), 7)) {
            DqnAgent agent = new DqnAgent(network);
            Random random = new Random(5);
            for (int i = 0; i < 30; i++) {
                var decision = agent.choose(scenario, observation, 1, random);
                assertTrue(decision.exploring());
                assertTrue(valid[decision.action().ordinal()]);
            }
            assertFalse(agent.choose(scenario, observation, 0, random).exploring());
        }
    }

    @Test
    void modelFileRestoresPredictionsAndRejectsWrongHeader() throws Exception {
        var scenario = TrainingScenarios.grounds().getFirst();
        float[] input = new ObservationEncoder().encode(scenario,
                new RescueEnvironment(scenario, RewardScheme.defaults(), 16).observation());
        var file = Files.createTempFile("neurorescue-model-test", ".nrdq");
        try {
            float[] expected;
            try (DqnNetwork network = new DqnNetwork(smallConfig(), 7)) {
                TrainingSession session = new TrainingSession(network, smallConfig(), 7);
                session.run(TrainingScenarios.grounds(), 4, ignored -> { });
                expected = network.values(input);
                network.save(file, 7, session.completedEpisodes());
            }
            DqnNetwork.LoadedModel loaded = DqnNetwork.load(file);
            try (DqnNetwork restored = loaded.network()) {
                assertEquals(4, loaded.completedEpisodes());
                assertArrayEquals(expected, restored.values(input), 1e-5f);
            }
            Files.write(file, new byte[]{0, 1, 2, 3});
            assertThrows(java.io.IOException.class, () -> DqnNetwork.load(file));
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
