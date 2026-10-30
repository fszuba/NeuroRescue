package it.farrlor.neurorescue.simulation;

import java.util.List;
import java.util.Random;

public final class Scenarios {
    private Scenarios() { }

    public static List<Scenario> builtIns() {
        return List.of(
                Scenario.fromAscii("First rescue",
                        "#########",
                        "#S......#",
                        "#.......#",
                        "#...V...#",
                        "#.......#",
                        "#......E#",
                        "#########"),
                Scenario.fromAscii("Blocked corridor",
                        "###########",
                        "#S....#...#",
                        "#..OO.#.V.#",
                        "#.....#...#",
                        "#.........#",
                        "#...H..E..#",
                        "###########"),
                Scenario.fromAscii("Two people",
                        "############",
                        "#S.....#...#",
                        "#..V...#...#",
                        "#..........#",
                        "#..O...H...#",
                        "#....V...E.#",
                        "############"),
                Scenario.fromAscii("Office building",
                        "############",
                        "#S...#.....#",
                        "#....#..V..#",
                        "#..........#",
                        "###.####.#.#",
                        "#...#..H...#",
                        "#.V.#..O.E.#",
                        "#..........#",
                        "############"),
                Scenario.fromAscii("Emergency facility",
                        "############",
                        "#S..#......#",
                        "#...#..H.V.#",
                        "#..........#",
                        "#.O.####...#",
                        "#...#...#..#",
                        "#.V...O.#E.#",
                        "#..........#",
                        "############"));
    }

    /** Generates a small map from an explicit seed and retries until all objectives have a safe route. */
    public static Scenario generated(long seed) {
        Random random = new Random(seed);
        for (int attempt = 0; attempt < 1_000; attempt++) {
            char[][] map = new char[7][9];
            for (int y = 0; y < map.length; y++) {
                for (int x = 0; x < map[y].length; x++) {
                    map[y][x] = x == 0 || y == 0 || x == 8 || y == 6 ? '#' : '.';
                }
            }
            map[1][1] = 'S';
            map[3][7] = 'V';
            map[5][7] = 'E';
            for (int i = 0; i < 8; i++) placeRandom(map, random, 'O');
            placeRandom(map, random, 'H');
            String[] rows = new String[map.length];
            for (int y = 0; y < map.length; y++) rows[y] = new String(map[y]);
            try {
                return Scenario.fromAscii("Generated · seed " + seed, rows);
            } catch (IllegalArgumentException unreachable) {
                // Reject maps that block the safe route and continue the seeded sequence.
            }
        }
        throw new IllegalStateException("Unable to generate a solvable scenario for seed " + seed);
    }

    private static void placeRandom(char[][] map, Random random, char tile) {
        while (true) {
            int x = 1 + random.nextInt(map[0].length - 2);
            int y = 1 + random.nextInt(map.length - 2);
            if (map[y][x] == '.') {
                map[y][x] = tile;
                return;
            }
        }
    }
}
