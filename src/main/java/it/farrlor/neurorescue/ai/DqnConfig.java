package it.farrlor.neurorescue.ai;

public record DqnConfig(float learningRate, float discount, float initialEpsilon,
                        float minimumEpsilon, float epsilonDecay, int replayCapacity,
                        int batchSize, int warmupSteps, int targetSyncSteps,
                        int maxEpisodeSteps, int hiddenUnits) {
    public DqnConfig {
        if (!Float.isFinite(learningRate) || learningRate <= 0 || learningRate > 1
                || !Float.isFinite(discount) || discount < 0 || discount > 1
                || !Float.isFinite(initialEpsilon) || initialEpsilon < 0 || initialEpsilon > 1
                || !Float.isFinite(minimumEpsilon) || minimumEpsilon < 0 || minimumEpsilon > initialEpsilon
                || !Float.isFinite(epsilonDecay) || epsilonDecay <= 0 || epsilonDecay > 1
                || replayCapacity < 1 || batchSize < 1 || batchSize > replayCapacity
                || warmupSteps < batchSize || targetSyncSteps < 1 || maxEpisodeSteps < 1
                || hiddenUnits < 2 || hiddenUnits > 512) {
            throw new IllegalArgumentException("Invalid DQN hyperparameters");
        }
    }

    public static DqnConfig defaults() {
        return new DqnConfig(0.001f, 0.95f, 1f, 0.05f, 0.995f,
                10_000, 32, 128, 200, 64, 64);
    }

    public float epsilonAtEpisode(int completedEpisodes) {
        if (completedEpisodes < 0) throw new IllegalArgumentException("Episode count must be nonnegative");
        return Math.max(minimumEpsilon, (float) (initialEpsilon * Math.pow(epsilonDecay, completedEpisodes)));
    }
}
