package it.farrlor.neurorescue.simulation;

public record Position(int x, int y) {
    public Position move(Action action) {
        return new Position(x + action.dx(), y + action.dy());
    }
}
