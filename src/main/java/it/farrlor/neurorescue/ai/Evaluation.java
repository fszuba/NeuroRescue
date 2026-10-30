package it.farrlor.neurorescue.ai;

import it.farrlor.neurorescue.baseline.RouteAgent;
import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.EpisodeStatus;
import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import it.farrlor.neurorescue.simulation.Scenario;

import java.util.List;
import java.util.Random;

/** Evaluation performs no optimizer step and always uses epsilon zero for neural policies. */
public final class Evaluation {
    private Evaluation() { }

    public static Summary neural(DqnNetwork network, List<Scenario> scenarios, int episodes, long seed, int maxSteps) {
        DqnAgent agent = new DqnAgent(network);
        return run(scenarios, episodes, seed, maxSteps,
                (scenario, observation, random) -> agent.choose(scenario, observation, 0, random).action());
    }

    public static Summary baseline(List<Scenario> scenarios, int episodes, long seed, int maxSteps) {
        RouteAgent baseline = new RouteAgent();
        return run(scenarios, episodes, seed, maxSteps,
                (scenario, observation, random) -> baseline.choose(scenario, observation));
    }

    public static Summary random(List<Scenario> scenarios, int episodes, long seed, int maxSteps) {
        return run(scenarios, episodes, seed, maxSteps, (scenario, observation, random) -> {
            boolean[] valid = ActionMask.valid(scenario, observation);
            int count = 0;
            for (boolean allowed : valid) if (allowed) count++;
            int choice = random.nextInt(count);
            for (int i = 0; i < valid.length; i++) if (valid[i] && choice-- == 0) return Action.values()[i];
            throw new IllegalStateException("No valid action");
        });
    }

    private static Summary run(List<Scenario> scenarios, int episodes, long seed, int maxSteps, Policy policy) {
        if (scenarios.isEmpty() || episodes < 1) throw new IllegalArgumentException("Evaluation needs maps and episodes");
        Random random = new Random(seed);
        int successes = 0, steps = 0, invalid = 0, hazards = 0;
        double reward = 0;
        for (int episode = 0; episode < episodes; episode++) {
            Scenario scenario = scenarios.get(episode % scenarios.size());
            RescueEnvironment environment = new RescueEnvironment(scenario, RewardScheme.defaults(), maxSteps);
            while (environment.observation().status() == EpisodeStatus.RUNNING) {
                environment.step(policy.choose(scenario, environment.observation(), random));
            }
            Observation result = environment.observation();
            if (result.status() == EpisodeStatus.SUCCESS) successes++;
            steps += result.steps();
            invalid += result.invalidActions();
            hazards += result.hazardEncounters();
            reward += result.totalReward();
        }
        return new Summary(episodes, successes, reward / episodes, steps / (double) episodes,
                invalid / (double) episodes, hazards / (double) episodes);
    }

    private interface Policy { Action choose(Scenario scenario, Observation observation, Random random); }

    public record Summary(int episodes, int successes, double averageReward, double averageSteps,
                          double averageInvalidActions, double averageHazards) {
        public double successRate() { return successes / (double) episodes; }
    }
}
