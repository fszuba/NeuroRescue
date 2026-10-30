package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.simulation.Scenario;

import java.util.List;

/** Small rescue tasks precede the larger built-in maps; all maps are safely solvable. */
public final class TrainingScenarios {
    private static final List<Scenario> GROUNDS = List.of(
                Scenario.fromAscii("Training grounds A", "#####", "#SVE#", "#...#", "#...#", "#####"),
                Scenario.fromAscii("Training grounds B", "#####", "#S..#", "#VE.#", "#...#", "#####"),
                Scenario.fromAscii("Training grounds C", "#####", "#..E#", "#SV.#", "#...#", "#####"));
    private static final List<Scenario> OBSTACLES = List.of(
                Scenario.fromAscii("Obstacle course A", "#######", "#S.O.V#", "#..O..#", "#.....#", "#..O.E#", "#######"),
                Scenario.fromAscii("Obstacle course B", "#######", "#S....#", "#.OO..#", "#...V.#", "#...E.#", "#######"));
    private static final List<Scenario> HAZARDS = List.of(
                Scenario.fromAscii("Hazard course A", "########", "#S..H.V#", "#..OO..#", "#......#", "#...E..#", "########"),
                Scenario.fromAscii("Hazard course B", "########", "#S.....#", "#..H.O.#", "#..O.V.#", "#....E.#", "########"));
    private static final List<Scenario> HELD_OUT = List.of(
                Scenario.fromAscii("Held-out grounds A", "#####", "#..S#", "#..V#", "#..E#", "#####"),
                Scenario.fromAscii("Held-out grounds B", "#####", "#E..#", "#V.S#", "#...#", "#####"));

    private TrainingScenarios() { }

    public static List<Scenario> grounds() { return GROUNDS; }
    public static List<Scenario> obstacleCourse() { return OBSTACLES; }
    public static List<Scenario> hazardousCourse() { return HAZARDS; }
    public static List<Scenario> heldOutGrounds() { return HELD_OUT; }
}
