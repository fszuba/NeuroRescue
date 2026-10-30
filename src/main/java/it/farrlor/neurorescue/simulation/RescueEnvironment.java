package it.farrlor.neurorescue.simulation;

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/** Single-thread-confined deterministic episode engine. The seed is reserved for future stochastic scenarios. */
public final class RescueEnvironment {
    private final Scenario scenario;
    private final RewardScheme rewards;
    private final int maxSteps;
    private final Set<Position> remainingVictims = new HashSet<>();
    private Position robot;
    private boolean carrying;
    private int rescued;
    private int steps;
    private int invalidActions;
    private int hazardEncounters;
    private double totalReward;
    private EpisodeStatus status;

    public RescueEnvironment(Scenario scenario, RewardScheme rewards, int maxSteps) {
        this.scenario = Objects.requireNonNull(scenario);
        this.rewards = Objects.requireNonNull(rewards);
        if (maxSteps < 1) throw new IllegalArgumentException("Maximum steps must be positive");
        this.maxSteps = maxSteps;
        reset(0);
    }

    public Scenario scenario() { return scenario; }
    public int maxSteps() { return maxSteps; }

    public Observation reset(long seed) {
        remainingVictims.clear();
        remainingVictims.addAll(scenario.victims());
        robot = scenario.start();
        carrying = false;
        rescued = 0;
        steps = 0;
        invalidActions = 0;
        hazardEncounters = 0;
        totalReward = 0;
        status = EpisodeStatus.RUNNING;
        return observation();
    }

    public Observation observation() {
        return new Observation(robot, remainingVictims, carrying, rescued, steps, invalidActions,
                hazardEncounters, totalReward, status);
    }

    public StepResult step(Action action) {
        Objects.requireNonNull(action, "action");
        if (status != EpisodeStatus.RUNNING) throw new IllegalStateException("Reset before stepping a finished episode");
        double reward = rewards.step();
        boolean valid = false;
        if (action.isMovement()) {
            Position destination = robot.move(action);
            if (scenario.contains(destination) && scenario.tileAt(destination).traversable()) {
                robot = destination;
                valid = true;
                if (scenario.tileAt(robot) == Tile.HAZARD) {
                    hazardEncounters++;
                    reward += rewards.hazard();
                    status = EpisodeStatus.FAILURE;
                }
            }
        } else if (action == Action.PICK_UP) {
            if (!carrying && remainingVictims.remove(robot)) {
                carrying = true;
                valid = true;
                reward += rewards.pickup();
            }
        } else if (carrying && scenario.exits().contains(robot)) {
            carrying = false;
            rescued++;
            valid = true;
            reward += rewards.delivery();
            if (remainingVictims.isEmpty()) {
                reward += rewards.completion();
                status = EpisodeStatus.SUCCESS;
            }
        }
        if (!valid) {
            invalidActions++;
            reward += rewards.invalidAction();
        }
        steps++;
        if (status == EpisodeStatus.RUNNING && steps >= maxSteps) status = EpisodeStatus.TRUNCATED;
        totalReward += reward;
        return new StepResult(observation(), reward, valid);
    }
}
