package it.farrlor.neurorescue.baseline;

import it.farrlor.neurorescue.simulation.EpisodeStatus;
import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import it.farrlor.neurorescue.simulation.Scenario;
import it.farrlor.neurorescue.simulation.Scenarios;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class RouteAgentTest {
    @Test
    void rescuesEveryoneOnEveryBuiltInMapWithoutInvalidActions() {
        RouteAgent agent = new RouteAgent();
        java.util.List<Scenario> scenarios = new java.util.ArrayList<>(Scenarios.builtIns());
        for (long seed = 0; seed < 25; seed++) scenarios.add(Scenarios.generated(seed));
        for (Scenario scenario : scenarios) {
            RescueEnvironment environment = new RescueEnvironment(scenario, RewardScheme.defaults(), 200);
            while (environment.observation().status() == EpisodeStatus.RUNNING) {
                environment.step(agent.choose(scenario, environment.observation()));
            }
            assertEquals(EpisodeStatus.SUCCESS, environment.observation().status(), scenario.name());
            assertEquals(0, environment.observation().invalidActions(), scenario.name());
            assertEquals(0, environment.observation().hazardEncounters(), scenario.name());
            assertEquals(scenario.victims().size(), environment.observation().rescued(), scenario.name());
        }
    }
}
