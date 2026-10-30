package it.farrlor.neurorescue.simulation;

public record StepResult(Observation observation, double reward, boolean valid) { }
