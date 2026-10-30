package it.farrlor.neurorescue.baseline;

import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Position;
import it.farrlor.neurorescue.simulation.Scenario;
import it.farrlor.neurorescue.simulation.Tile;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/** Shortest safe route under complete map knowledge; serves as a transparent non-neural baseline. */
public final class RouteAgent {
    private static final Action[] MOVES = {Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT};

    public Action choose(Scenario scenario, Observation observation) {
        if (observation.carrying()) {
            if (scenario.exits().contains(observation.robot())) return Action.DROP_OFF;
            return firstMove(scenario, observation.robot(), scenario.exits());
        }
        if (observation.victims().contains(observation.robot())) return Action.PICK_UP;
        return firstMove(scenario, observation.robot(), observation.victims());
    }

    private Action firstMove(Scenario scenario, Position start, Set<Position> goals) {
        ArrayDeque<Position> queue = new ArrayDeque<>();
        Map<Position, Action> firstActions = new HashMap<>();
        queue.add(start);
        firstActions.put(start, null);
        while (!queue.isEmpty()) {
            Position current = queue.removeFirst();
            for (Action move : MOVES) {
                Position next = current.move(move);
                if (!scenario.contains(next) || !scenario.tileAt(next).traversable()
                        || scenario.tileAt(next) == Tile.HAZARD || firstActions.containsKey(next)) continue;
                Action first = current.equals(start) ? move : firstActions.get(current);
                if (goals.contains(next)) return first;
                firstActions.put(next, first);
                queue.addLast(next);
            }
        }
        throw new IllegalStateException("No safe route to objective in validated scenario");
    }
}
