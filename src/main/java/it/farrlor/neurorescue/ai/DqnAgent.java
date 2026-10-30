package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Scenario;

import java.util.Random;

/** Applies the same mask to exploratory and greedy decisions. */
public final class DqnAgent {
    private final DqnNetwork network;
    private final ObservationEncoder encoder = new ObservationEncoder();

    public DqnAgent(DqnNetwork network) { this.network = network; }

    public Decision choose(Scenario scenario, Observation observation, float epsilon, Random random) {
        if (epsilon < 0 || epsilon > 1) throw new IllegalArgumentException("Epsilon must be between 0 and 1");
        boolean[] valid = ActionMask.valid(scenario, observation);
        float[] values = network.values(encoder.encode(scenario, observation));
        boolean exploring = random.nextFloat() < epsilon;
        int selected;
        if (exploring) {
            int count = 0;
            for (boolean allowed : valid) if (allowed) count++;
            int choice = random.nextInt(count);
            selected = 0;
            for (int i = 0; i < valid.length; i++) {
                if (valid[i] && choice-- == 0) { selected = i; break; }
            }
        } else {
            selected = ActionMask.greedy(values, valid);
        }
        return new Decision(Action.values()[selected], values, exploring);
    }

    public record Decision(Action action, float[] qValues, boolean exploring) {
        public Decision { qValues = qValues.clone(); }
        @Override public float[] qValues() { return qValues.clone(); }
    }
}
