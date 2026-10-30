package it.farrlor.neurorescue.simulation;

public record RewardScheme(double step, double invalidAction, double pickup, double delivery,
                           double completion, double hazard) {
    public RewardScheme {
        if (!Double.isFinite(step) || !Double.isFinite(invalidAction) || !Double.isFinite(pickup)
                || !Double.isFinite(delivery) || !Double.isFinite(completion) || !Double.isFinite(hazard)
                || step > 0 || invalidAction > 0 || hazard > 0 || pickup < 0 || delivery < 0 || completion < 0) {
            throw new IllegalArgumentException("Invalid reward scheme");
        }
    }

    public static RewardScheme defaults() {
        return new RewardScheme(-0.1, -1.0, 2.0, 15.0, 20.0, -10.0);
    }
}
