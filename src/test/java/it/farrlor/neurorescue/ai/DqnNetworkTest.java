package it.farrlor.neurorescue.ai;

import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DqnNetworkTest {
    @Test
    void bellmanTargetBootstrapsTruncationButNotTermination() {
        float[] q = {2, 99, 4, -1, 0, 0};
        boolean[] valid = {true, false, true, false, false, false};
        assertEquals(3.5f, DqnNetwork.bellmanTarget(1.5f, false, q, 0, valid, 0.5f));
        assertEquals(1.5f, DqnNetwork.bellmanTarget(1.5f, true, q, 0, valid, 0.5f));
    }

    @Test
    void optimizationChangesOnlineValuesUntilTargetIsSynchronized() {
        DqnConfig config = new DqnConfig(0.01f, 0.9f, 1, 0.1f, 0.99f,
                64, 8, 8, 1_000, 20, 16);
        try (DqnNetwork network = new DqnNetwork(config, 42)) {
            float[] state = new float[ObservationEncoder.SIZE];
            float[] before = network.values(state);
            assertArrayEquals(before, network.targetValues(state), 1e-5f);
            Experience transition = new Experience(state, 0, 3f, state, new boolean[6], true, false);
            List<Experience> batch = Collections.nCopies(8, transition);
            float initialLoss = network.trainBatch(batch);
            float finalLoss = initialLoss;
            for (int i = 0; i < 50; i++) finalLoss = network.trainBatch(batch);
            float[] after = network.values(state);
            assertEquals(6, after.length);
            assertNotEquals(before[0], after[0]);
            assertArrayEquals(before, network.targetValues(state), 1e-5f);
            assertTrue(finalLoss < initialLoss, "DQN optimization should reduce a fixed terminal target loss");
            network.synchronizeTarget();
            assertArrayEquals(after, network.targetValues(state), 1e-5f);
        }
    }
}
