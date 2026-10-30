package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Action;

public record Experience(float[] state, int action, float reward, float[] nextState,
                         boolean[] nextValidActions, boolean terminated, boolean truncated) {
    public Experience {
        if (state == null || state.length != ObservationEncoder.SIZE || nextState == null
                || nextState.length != ObservationEncoder.SIZE || nextValidActions == null
                || nextValidActions.length != Action.values().length || action < 0
                || action >= Action.values().length || !Float.isFinite(reward)
                || (terminated && truncated)) {
            throw new IllegalArgumentException("Invalid DQN experience");
        }
        state = state.clone();
        nextState = nextState.clone();
        nextValidActions = nextValidActions.clone();
    }

    @Override public float[] state() { return state.clone(); }
    @Override public float[] nextState() { return nextState.clone(); }
    @Override public boolean[] nextValidActions() { return nextValidActions.clone(); }
}
