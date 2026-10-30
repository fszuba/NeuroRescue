package it.farrlor.neurorescue;

import java.util.ArrayDeque;
import java.util.List;

/** The samples shown by a chart, indexed by the episode that produced each measurement. */
final class TrainingChartData {
    record Sample(int episode, double value, double average) { }

    private final int capacity;
    private final ArrayDeque<Sample> samples = new ArrayDeque<>();

    TrainingChartData(int capacity) {
        if (capacity < 1) throw new IllegalArgumentException("Chart capacity must be positive");
        this.capacity = capacity;
    }

    void add(int episode, double value, double average) {
        if (!Double.isFinite(value)) return;
        if (episode < 1 || (!samples.isEmpty() && episode <= samples.getLast().episode())) {
            throw new IllegalArgumentException("Chart episodes must increase");
        }
        samples.addLast(new Sample(episode, value, Double.isFinite(average) ? average : Double.NaN));
        if (samples.size() > capacity) samples.removeFirst();
    }

    List<Sample> samples() {
        return List.copyOf(samples);
    }

    double horizontalPosition(int episode) {
        if (samples.isEmpty()) throw new IllegalStateException("Chart has no samples");
        int first = samples.getFirst().episode();
        int last = samples.getLast().episode();
        return first == last ? 0.5 : (episode - first) / (double) (last - first);
    }

    void clear() {
        samples.clear();
    }
}
