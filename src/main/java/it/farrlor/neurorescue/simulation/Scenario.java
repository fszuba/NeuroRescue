package it.farrlor.neurorescue.simulation;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Immutable scenario. ASCII maps use # walls, O obstacles, E exits, V victims, H hazards, and S start. */
public final class Scenario {
    private final String name;
    private final Tile[][] tiles;
    private final Position start;
    private final Set<Position> victims;
    private final Set<Position> exits;

    private Scenario(String name, Tile[][] tiles, Position start, Set<Position> victims, Set<Position> exits) {
        this.name = name;
        this.tiles = tiles;
        this.start = start;
        this.victims = Set.copyOf(victims);
        this.exits = Set.copyOf(exits);
    }

    public static Scenario fromAscii(String name, String... rows) {
        Objects.requireNonNull(name, "name");
        if (name.isBlank() || rows == null || rows.length < 3 || rows.length > 40 || rows[0] == null
                || rows[0].length() < 3 || rows[0].length() > 40) {
            throw new IllegalArgumentException("Scenario needs a name and a 3–40 tile rectangular map");
        }
        int width = rows[0].length();
        Tile[][] tiles = new Tile[rows.length][width];
        Set<Position> victims = new HashSet<>();
        Set<Position> exits = new HashSet<>();
        Position start = null;
        for (int y = 0; y < rows.length; y++) {
            if (rows[y] == null || rows[y].length() != width) {
                throw new IllegalArgumentException("Every scenario row must have the same width");
            }
            for (int x = 0; x < width; x++) {
                Position position = new Position(x, y);
                tiles[y][x] = switch (rows[y].charAt(x)) {
                    case '#' -> Tile.WALL;
                    case 'O' -> Tile.OBSTACLE;
                    case 'H' -> Tile.HAZARD;
                    case 'E' -> { exits.add(position); yield Tile.EXIT; }
                    case 'V' -> { victims.add(position); yield Tile.FLOOR; }
                    case 'S' -> {
                        if (start != null) throw new IllegalArgumentException("Scenario must have one start");
                        start = position;
                        yield Tile.FLOOR;
                    }
                    case '.', ' ' -> Tile.FLOOR;
                    default -> throw new IllegalArgumentException("Unknown tile at " + position);
                };
            }
        }
        if (start == null || exits.isEmpty() || victims.isEmpty()) {
            throw new IllegalArgumentException("Scenario needs one start, an exit, and a victim");
        }
        Scenario scenario = new Scenario(name, tiles, start, victims, exits);
        if (!scenario.allObjectivesReachable()) {
            throw new IllegalArgumentException("Victims and exits must be reachable without hazards");
        }
        return scenario;
    }

    private boolean allObjectivesReachable() {
        Set<Position> reached = new HashSet<>();
        java.util.ArrayDeque<Position> queue = new java.util.ArrayDeque<>();
        queue.add(start);
        reached.add(start);
        while (!queue.isEmpty()) {
            Position current = queue.removeFirst();
            for (Action action : List.of(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)) {
                Position next = current.move(action);
                if (contains(next) && tileAt(next).traversable() && tileAt(next) != Tile.HAZARD && reached.add(next)) {
                    queue.addLast(next);
                }
            }
        }
        return reached.containsAll(victims) && reached.containsAll(exits);
    }

    public String name() { return name; }
    public int width() { return tiles[0].length; }
    public int height() { return tiles.length; }
    public Position start() { return start; }
    public Set<Position> victims() { return victims; }
    public Set<Position> exits() { return exits; }
    public boolean contains(Position position) {
        return position.x() >= 0 && position.y() >= 0 && position.x() < width() && position.y() < height();
    }
    public Tile tileAt(Position position) {
        if (!contains(position)) throw new IllegalArgumentException("Position outside scenario: " + position);
        return tiles[position.y()][position.x()];
    }
}
