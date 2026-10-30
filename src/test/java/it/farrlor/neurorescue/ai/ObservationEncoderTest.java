package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import it.farrlor.neurorescue.simulation.Scenario;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ObservationEncoderTest {
    @Test
    void encodesMapAndCarryingStateWithoutChangingLength() {
        Scenario scenario = Scenario.fromAscii("encoder", "#####", "#SVE#", "#...#", "#####");
        RescueEnvironment environment = new RescueEnvironment(scenario, RewardScheme.defaults(), 20);
        ObservationEncoder encoder = new ObservationEncoder();
        float[] initial = encoder.encode(scenario, environment.observation());
        assertEquals(ObservationEncoder.SIZE, initial.length);
        assertEquals(1, initial[1 * ObservationEncoder.WIDTH + 0]);
        assertEquals(1, initial[4 * 144 + 1 * 12 + 2]);
        assertEquals(1, initial[5 * 144 + 1 * 12 + 1]);
        assertEquals(0, initial[6 * 144]);

        environment.step(Action.RIGHT);
        environment.step(Action.PICK_UP);
        float[] carrying = encoder.encode(scenario, environment.observation());
        assertEquals(0, carrying[4 * 144 + 1 * 12 + 2]);
        assertEquals(1, carrying[6 * 144]);
        assertEquals(ObservationEncoder.SIZE, carrying.length);
    }

    @Test
    void rejectsOversizedMapsInsteadOfResizingThem() {
        Scenario scenario = Scenario.fromAscii("wide", "##############", "#S.V.E.......#", "##############");
        RescueEnvironment environment = new RescueEnvironment(scenario, RewardScheme.defaults(), 20);
        assertThrows(IllegalArgumentException.class,
                () -> new ObservationEncoder().encode(scenario, environment.observation()));
    }

    @Test
    void actionMaskMatchesEnvironmentRules() {
        Scenario scenario = Scenario.fromAscii("mask", "#####", "#SVE#", "#...#", "#####");
        RescueEnvironment environment = new RescueEnvironment(scenario, RewardScheme.defaults(), 20);
        boolean[] start = ActionMask.valid(scenario, environment.observation());
        assertFalse(start[Action.UP.ordinal()]);
        assertTrue(start[Action.RIGHT.ordinal()]);
        assertFalse(start[Action.PICK_UP.ordinal()]);
        environment.step(Action.RIGHT);
        assertTrue(ActionMask.valid(scenario, environment.observation())[Action.PICK_UP.ordinal()]);
        environment.step(Action.PICK_UP);
        assertFalse(ActionMask.valid(scenario, environment.observation())[Action.PICK_UP.ordinal()]);
        assertEquals(Action.RIGHT.ordinal(), ActionMask.greedy(new float[]{0, 0, 0, 3, 10, 10},
                ActionMask.valid(scenario, environment.observation())));
    }
}
