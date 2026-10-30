package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.EpisodeStatus;

public record EpisodeMetrics(EpisodeStatus status, double reward, int steps,
                             int invalidActions, int hazardEncounters, double meanLoss) {
    public boolean success() { return status == EpisodeStatus.SUCCESS; }
}
