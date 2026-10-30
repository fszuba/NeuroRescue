package it.farrlor.neurorescue.ai;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/** Run explicitly with -Dneurorescue.experiment=true; this is deliberately outside the quick suite. */
class LearningExperimentTest {
    @Test
    @EnabledIfSystemProperty(named = "neurorescue.experiment", matches = "true")
    void controlledRescueLearning() throws Exception {
        long seed = 17;
        DqnConfig defaults = DqnConfig.defaults();
        DqnConfig config = new DqnConfig(defaults.learningRate(), defaults.discount(),
                defaults.initialEpsilon(), defaults.minimumEpsilon(), 0.985f,
                3_000, 32, 64, 100, 32, 64);
        Path file = Files.createTempFile("neurorescue-experiment", ".nrdq");
        try (DqnNetwork network = new DqnNetwork(config, seed)) {
            var maps = TrainingScenarios.grounds();
            var initial = Evaluation.neural(network, maps, 30, 23, config.maxEpisodeSteps());
            var baseline = Evaluation.baseline(maps, 30, 23, config.maxEpisodeSteps());
            var random = Evaluation.random(maps, 30, 23, config.maxEpisodeSteps());
            System.out.println("Initial=" + initial + " baseline=" + baseline + " random=" + random);
            TrainingSession session = new TrainingSession(network, config, seed);
            session.run(maps, 360, progress -> {
                if (progress.completedEpisodes() % 60 == 0) {
                    var evaluated = Evaluation.neural(network, maps, 30, 23, config.maxEpisodeSteps());
                    System.out.println("episode=" + progress.completedEpisodes() + " epsilon=" + progress.epsilon()
                            + " trainingReward=" + progress.rollingReward() + " trainingSuccess="
                            + progress.rollingTrainingSuccess() + " loss=" + progress.episode().meanLoss()
                            + " evaluation=" + evaluated);
                }
            });
            var trained = Evaluation.neural(network, maps, 30, 23, config.maxEpisodeSteps());
            var heldOut = Evaluation.neural(network, TrainingScenarios.heldOutGrounds(), 20, 23, config.maxEpisodeSteps());
            System.out.println("Final=" + trained + " heldOut=" + heldOut);
            network.save(file, seed, session.completedEpisodes());
            float[] input = new ObservationEncoder().encode(maps.getFirst(),
                    new it.farrlor.neurorescue.simulation.RescueEnvironment(maps.getFirst(),
                            it.farrlor.neurorescue.simulation.RewardScheme.defaults(), 32).observation());
            float[] expected = network.values(input);
            DqnNetwork.LoadedModel loaded = DqnNetwork.load(file);
            try (DqnNetwork restored = loaded.network()) {
                assertEquals(session.completedEpisodes(), loaded.completedEpisodes());
                assertArrayEquals(expected, restored.values(input), 1e-5f);
            }
            assertTrue(trained.successRate() > initial.successRate(), "Training must improve controlled-map success");
        } finally {
            Files.deleteIfExists(file);
        }
    }
}
