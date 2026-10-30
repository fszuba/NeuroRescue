package it.farrlor.neurorescue.simulation;

public enum Action {
    UP(0, -1), DOWN(0, 1), LEFT(-1, 0), RIGHT(1, 0), PICK_UP(0, 0), DROP_OFF(0, 0);

    private final int dx;
    private final int dy;

    Action(int dx, int dy) {
        this.dx = dx;
        this.dy = dy;
    }

    public int dx() { return dx; }
    public int dy() { return dy; }
    public boolean isMovement() { return this != PICK_UP && this != DROP_OFF; }
}
