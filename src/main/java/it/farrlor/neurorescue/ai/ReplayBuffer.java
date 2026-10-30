package it.farrlor.neurorescue.ai;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/** Bounded ring buffer. A sample contains distinct transitions. */
public final class ReplayBuffer {
    private final Experience[] entries;
    private int size;
    private int next;

    public ReplayBuffer(int capacity) {
        if (capacity < 1) throw new IllegalArgumentException("Replay capacity must be positive");
        entries = new Experience[capacity];
    }

    public int size() { return size; }
    public int capacity() { return entries.length; }

    public void add(Experience experience) {
        entries[next] = Objects.requireNonNull(experience);
        next = (next + 1) % entries.length;
        if (size < entries.length) size++;
    }

    public List<Experience> sample(int count, Random random) {
        Objects.requireNonNull(random);
        if (count < 1 || count > size) throw new IllegalArgumentException("Sample size outside replay range");
        List<Experience> sample = new ArrayList<>(count);
        Set<Integer> selected = new HashSet<>();
        while (sample.size() < count) {
            int index = random.nextInt(size);
            if (selected.add(index)) sample.add(entries[index]);
        }
        return List.copyOf(sample);
    }
}
