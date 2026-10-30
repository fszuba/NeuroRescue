package it.farrlor.neurorescue.simulation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ScenarioTest {
    @Test
    void rejectsUnreachableObjectives() {
        assertThrows(IllegalArgumentException.class, () -> Scenario.fromAscii("blocked",
                "#######",
                "#S#V.E#",
                "#######"));
    }

    @Test
    void rejectsRaggedMapAndDuplicateStart() {
        assertThrows(IllegalArgumentException.class, () -> Scenario.fromAscii("ragged", "###", "#S#", "#VE##"));
        assertThrows(IllegalArgumentException.class, () -> Scenario.fromAscii("duplicate", "#####", "#SSV#", "#..E#", "#####"));
    }

    @Test
    void builtInMapsHaveReachableObjectives() {
        assertEquals(5, Scenarios.builtIns().size());
        for (Scenario scenario : Scenarios.builtIns()) {
            assertFalse(scenario.victims().isEmpty());
            assertFalse(scenario.exits().isEmpty());
        }
    }

    @Test
    void generatedMapsAreReproducibleAndSolvable() {
        Scenario first = Scenarios.generated(8794);
        Scenario repeated = Scenarios.generated(8794);
        for (int y = 0; y < first.height(); y++) {
            for (int x = 0; x < first.width(); x++) {
                Position position = new Position(x, y);
                assertEquals(first.tileAt(position), repeated.tileAt(position));
            }
        }
        assertEquals(first.victims(), repeated.victims());
        assertEquals(first.exits(), repeated.exits());
    }
}
