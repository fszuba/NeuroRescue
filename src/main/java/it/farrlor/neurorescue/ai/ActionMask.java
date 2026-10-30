package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Position;
import it.farrlor.neurorescue.simulation.Scenario;

/** Masks impossible actions; entering a traversable hazard remains possible and must be learned as dangerous. */
public final class ActionMask {
    private ActionMask() { }

    public static boolean[] valid(Scenario scenario, Observation observation) {
        boolean[] valid = new boolean[Action.values().length];
        for (Action action : Action.values()) {
            if (action.isMovement()) {
                Position destination = observation.robot().move(action);
                valid[action.ordinal()] = scenario.contains(destination)
                        && scenario.tileAt(destination).traversable();
            } else if (action == Action.PICK_UP) {
                valid[action.ordinal()] = !observation.carrying()
                        && observation.victims().contains(observation.robot());
            } else {
                valid[action.ordinal()] = observation.carrying()
                        && scenario.exits().contains(observation.robot());
            }
        }
        return valid;
    }

    public static int greedy(float[] values, boolean[] valid) {
        if (values.length != Action.values().length || valid.length != values.length) {
            throw new IllegalArgumentException("Q-values and action mask must match the six-action space");
        }
        int best = -1;
        for (int i = 0; i < values.length; i++) {
            if (!valid[i]) continue;
            if (!Float.isFinite(values[i])) throw new IllegalArgumentException("Non-finite Q-value");
            if (best < 0 || values[i] > values[best]) best = i;
        }
        if (best < 0) throw new IllegalArgumentException("No valid action in a running state");
        return best;
    }
}
