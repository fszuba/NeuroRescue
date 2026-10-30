package it.farrlor.neurorescue.simulation;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class RescueEnvironmentTest {
    private static final Scenario SIMPLE = Scenario.fromAscii("test",
            "#####",
            "#SVE#",
            "#...#",
            "#####");

    @Test
    void resetAndActionSequenceAreDeterministic() {
        RescueEnvironment environment = new RescueEnvironment(SIMPLE, RewardScheme.defaults(), 20);
        List<Action> actions = List.of(Action.RIGHT, Action.PICK_UP, Action.RIGHT, Action.DROP_OFF);
        Observation initial = environment.reset(42);
        List<StepResult> first = actions.stream().map(environment::step).toList();
        assertEquals(initial, environment.reset(42));
        List<StepResult> second = actions.stream().map(environment::step).toList();
        assertEquals(first, second);
        assertEquals(EpisodeStatus.SUCCESS, second.getLast().observation().status());
    }

    @Test
    void invalidMovementKeepsPositionAndChargesPenalty() {
        RescueEnvironment environment = new RescueEnvironment(SIMPLE, RewardScheme.defaults(), 20);
        StepResult result = environment.step(Action.LEFT);
        assertFalse(result.valid());
        assertEquals(SIMPLE.start(), result.observation().robot());
        assertEquals(-1.1, result.reward(), 1e-10);
        assertEquals(1, result.observation().invalidActions());
    }

    @Test
    void pickupAndDeliveryRequireCorrectTileAndState() {
        RescueEnvironment environment = new RescueEnvironment(SIMPLE, RewardScheme.defaults(), 20);
        assertFalse(environment.step(Action.PICK_UP).valid());
        environment.step(Action.RIGHT);
        StepResult pickup = environment.step(Action.PICK_UP);
        assertTrue(pickup.valid());
        assertTrue(pickup.observation().carrying());
        assertFalse(pickup.observation().victims().contains(new Position(2, 1)));
        assertFalse(environment.step(Action.PICK_UP).valid());
        environment.step(Action.RIGHT);
        StepResult delivery = environment.step(Action.DROP_OFF);
        assertTrue(delivery.valid());
        assertEquals(34.9, delivery.reward(), 1e-10);
        assertEquals(1, delivery.observation().rescued());
        assertEquals(EpisodeStatus.SUCCESS, delivery.observation().status());
        assertThrows(IllegalStateException.class, () -> environment.step(Action.UP));
    }

    @Test
    void hazardCausesFailureAndLimitCausesTruncation() {
        Scenario hazard = Scenario.fromAscii("hazard",
                "######",
                "#SHVE#",
                "#....#",
                "######");
        RescueEnvironment environment = new RescueEnvironment(hazard, RewardScheme.defaults(), 10);
        StepResult failure = environment.step(Action.RIGHT);
        assertEquals(EpisodeStatus.FAILURE, failure.observation().status());
        assertEquals(1, failure.observation().hazardEncounters());
        assertEquals(-10.1, failure.reward(), 1e-10);

        RescueEnvironment shortEpisode = new RescueEnvironment(SIMPLE, RewardScheme.defaults(), 1);
        StepResult truncated = shortEpisode.step(Action.RIGHT);
        assertEquals(EpisodeStatus.TRUNCATED, truncated.observation().status());
    }

    @Test
    void snapshotsDoNotChangeWhenEpisodeAdvances() {
        RescueEnvironment environment = new RescueEnvironment(SIMPLE, RewardScheme.defaults(), 20);
        Observation before = environment.observation();
        environment.step(Action.RIGHT);
        environment.step(Action.PICK_UP);
        assertTrue(before.victims().contains(new Position(2, 1)));
        assertThrows(UnsupportedOperationException.class, () -> before.victims().clear());
    }
}
