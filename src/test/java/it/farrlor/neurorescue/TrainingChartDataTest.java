package it.farrlor.neurorescue;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TrainingChartDataTest {
    @Test
    void irregularEpisodesUseMeasuredPositions() {
        TrainingChartData data = new TrainingChartData(10);
        data.add(1, -2, -2);
        data.add(20, 10, 4);
        data.add(100, 16, 8);
        assertEquals(19.0 / 99.0, data.horizontalPosition(20), 1e-9);
        assertEquals(4, data.samples().get(1).average());
        assertEquals(100, data.samples().getLast().episode());
    }

    @Test
    void missingLossIsSkippedAndHistoryIsBounded() {
        TrainingChartData data = new TrainingChartData(2);
        data.add(1, Double.NaN, Double.NaN);
        assertTrue(data.samples().isEmpty());
        data.add(5, 2, Double.NaN);
        data.add(20, 1, Double.NaN);
        data.add(40, 0.5, Double.NaN);
        assertEquals(2, data.samples().size());
        assertEquals(20, data.samples().getFirst().episode());
        assertTrue(Double.isNaN(data.samples().getFirst().average()));
        assertEquals(1, data.horizontalPosition(40));
    }

    @Test
    void resetClearsSamplesAndAllowsNewEpisodeSequence() {
        TrainingChartData data = new TrainingChartData(3);
        data.add(10, 1, 1);
        assertThrows(IllegalArgumentException.class, () -> data.add(10, 2, 2));
        data.clear();
        assertTrue(data.samples().isEmpty());
        data.add(1, 3, 3);
        assertEquals(0.5, data.horizontalPosition(1));
    }
}
