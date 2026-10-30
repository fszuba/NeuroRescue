package it.farrlor.neurorescue.simulation;

public enum Tile {
    FLOOR, WALL, OBSTACLE, EXIT, HAZARD;

    public boolean traversable() {
        return this != WALL && this != OBSTACLE;
    }
}
