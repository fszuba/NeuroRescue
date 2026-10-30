package it.farrlor.neurorescue;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TrainingStatusTest {
    @Test
    void completionCancellationAndPauseHaveDistinctLabels() {
        assertEquals("Completed · 600 / 600 episodes", TrainingStatus.COMPLETED.label(600, 600));
        assertEquals("Stopped · 245 / 600 episodes", TrainingStatus.STOPPED.label(245, 600));
        assertEquals("Paused · 245 / 600 episodes", TrainingStatus.PAUSED.label(245, 600));
        assertEquals("Training failed · See error details", TrainingStatus.FAILED.label(245, 600));
    }
}
