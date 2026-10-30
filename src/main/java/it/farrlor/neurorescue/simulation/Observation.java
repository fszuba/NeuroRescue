package it.farrlor.neurorescue.simulation;

import java.util.Set;

/** Immutable episode state shared by simulation, baseline, and neural policies. */
public record Observation(Position robot, Set<Position> victims, boolean carrying, int rescued,
                          int steps, int invalidActions, int hazardEncounters, double totalReward,
                          EpisodeStatus status) {
    public Observation {
        victims = Set.copyOf(victims);
    }
}
