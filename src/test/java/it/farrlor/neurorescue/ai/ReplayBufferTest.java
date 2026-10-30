package it.farrlor.neurorescue.ai;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

class ReplayBufferTest {
    private Experience experience(float reward) {
        return new Experience(new float[ObservationEncoder.SIZE], 0, reward,
                new float[ObservationEncoder.SIZE], new boolean[6], true, false);
    }

    @Test
    void overwritesOldestAndSamplesDistinctTransitions() {
        ReplayBuffer replay = new ReplayBuffer(3);
        for (int i = 0; i < 5; i++) replay.add(experience(i));
        assertEquals(3, replay.size());
        List<Experience> sampled = replay.sample(3, new Random(42));
        assertEquals(3, sampled.size());
        assertEquals(3, sampled.stream().map(Experience::reward).distinct().count());
        assertTrue(sampled.stream().allMatch(item -> item.reward() >= 2));
        assertThrows(IllegalArgumentException.class, () -> replay.sample(4, new Random()));
    }

    @Test
    void experienceOwnsItsArrays() {
        float[] state = new float[ObservationEncoder.SIZE];
        Experience item = new Experience(state, 0, 1, state, new boolean[6], true, false);
        state[0] = 9;
        assertEquals(0, item.state()[0]);
        item.state()[0] = 7;
        assertEquals(0, item.state()[0]);
    }
}
