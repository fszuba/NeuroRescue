package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Position;
import it.farrlor.neurorescue.simulation.Scenario;
import it.farrlor.neurorescue.simulation.Tile;

import java.util.Comparator;
import java.util.Objects;

/** Six 12x12 binary map channels and seven normalized agent/goal features. */
public final class ObservationEncoder {
    public static final int WIDTH = 12;
    public static final int HEIGHT = 12;
    public static final int CHANNELS = 6;
    public static final int SIZE = WIDTH * HEIGHT * CHANNELS + 7;
    public static final int VERSION = 1;
    private static final int PLANE = WIDTH * HEIGHT;

    public float[] encode(Scenario scenario, Observation observation) {
        Objects.requireNonNull(scenario);
        Objects.requireNonNull(observation);
        if (scenario.width() > WIDTH || scenario.height() > HEIGHT) {
            throw new IllegalArgumentException("DQN supports maps up to 12×12 tiles; selected map is "
                    + scenario.width() + "×" + scenario.height());
        }
        float[] encoded = new float[SIZE];
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                Position position = new Position(x, y);
                int cell = y * WIDTH + x;
                if (!scenario.contains(position)) {
                    encoded[cell] = 1;
                    continue;
                }
                Tile tile = scenario.tileAt(position);
                switch (tile) {
                    case WALL -> encoded[cell] = 1;
                    case OBSTACLE -> encoded[PLANE + cell] = 1;
                    case HAZARD -> encoded[2 * PLANE + cell] = 1;
                    case EXIT -> encoded[3 * PLANE + cell] = 1;
                    case FLOOR -> { }
                }
                if (observation.victims().contains(position)) encoded[4 * PLANE + cell] = 1;
                if (observation.robot().equals(position)) encoded[5 * PLANE + cell] = 1;
            }
        }
        Position robot = observation.robot();
        Comparator<Position> byDistance = Comparator
                .comparingInt((Position p) -> Math.abs(p.x() - robot.x()) + Math.abs(p.y() - robot.y()))
                .thenComparingInt(Position::y).thenComparingInt(Position::x);
        Position victim = observation.victims().stream()
                .min(byDistance)
                .orElse(robot);
        Position exit = scenario.exits().stream()
                .min(byDistance)
                .orElseThrow();
        int offset = CHANNELS * PLANE;
        encoded[offset] = observation.carrying() ? 1 : 0;
        encoded[offset + 1] = robot.x() / (float) (WIDTH - 1);
        encoded[offset + 2] = robot.y() / (float) (HEIGHT - 1);
        encoded[offset + 3] = (victim.x() - robot.x()) / (float) (WIDTH - 1);
        encoded[offset + 4] = (victim.y() - robot.y()) / (float) (HEIGHT - 1);
        encoded[offset + 5] = (exit.x() - robot.x()) / (float) (WIDTH - 1);
        encoded[offset + 6] = (exit.y() - robot.y()) / (float) (HEIGHT - 1);
        return encoded;
    }
}
