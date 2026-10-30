package it.farrlor.neurorescue;

import it.farrlor.neurorescue.simulation.Observation;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;

import java.util.Locale;

final class MissionPanel extends VBox {
    final TextField seed = new TextField("2026");
    final Slider speed = new Slider(0.5, 3.0, 1.0);
    final Button loadModel = new Button("Load model…");
    final Button generate = new Button("Generate");

    private final Label status = new Label();
    private final Label steps = new Label();
    private final Label reward = new Label();
    private final Label rescued = new Label();
    private final Label carrying = new Label();
    private final Label invalid = new Label();
    private final Label lastAction = new Label("—");
    private final Label modelStatus = new Label("No trained model");
    private final Label qValues = new Label("Select a neural agent to see Q-values.");

    MissionPanel() {
        super(15);
        getStyleClass().add("side-panel");
        Label title = new Label("MISSION CONTROL");
        title.getStyleClass().add("panel-title");
        Label description = new Label("Live episode · selected policy");
        description.getStyleClass().add("muted");
        description.setWrapText(true);
        VBox metrics = new VBox(12, metric("STATUS", status), metric("STEPS", steps),
                metric("TOTAL REWARD", reward), metric("PEOPLE RESCUED", rescued),
                metric("CARRYING", carrying), metric("INVALID ACTIONS", invalid),
                metric("LAST ACTION", lastAction), metric("MODEL", modelStatus));
        metrics.getStyleClass().add("metric-list");
        Label inspectorTitle = new Label("NEURAL INSPECTOR");
        inspectorTitle.getStyleClass().add("panel-title");
        qValues.setWrapText(true);
        qValues.getStyleClass().add("q-values");
        seed.setPrefColumnCount(7);
        seed.setTooltip(new Tooltip("Fixed seed for a reproducible generated map"));
        HBox generator = new HBox(7, seed, generate);
        generator.setAlignment(Pos.CENTER_LEFT);
        speed.setShowTickLabels(true);
        speed.setMajorTickUnit(0.5);
        speed.setBlockIncrement(0.25);
        Tooltip.install(speed, new Tooltip("Playback speed multiplier"));
        Label speedLabel = new Label("PLAYBACK SPEED");
        speedLabel.getStyleClass().add("eyebrow");
        Label mapSeedLabel = new Label("MAP SEED");
        mapSeedLabel.getStyleClass().add("eyebrow");
        Label manual = new Label("Manual: arrow keys or WASD · P pickup · E drop");
        manual.getStyleClass().add("muted");
        manual.setWrapText(true);
        getChildren().addAll(title, description, metrics, inspectorTitle, qValues,
                loadModel, mapSeedLabel, generator, speedLabel, speed, manual);
    }

    private static VBox metric(String name, Label value) {
        Label label = new Label(name);
        label.getStyleClass().add("eyebrow");
        value.getStyleClass().add("metric-value");
        return new VBox(3, label, value);
    }

    void show(Observation observation, int maxSteps, int victimCount, int modelEpisodes) {
        status.setText(switch (observation.status()) {
            case RUNNING -> "Active";
            case SUCCESS -> "All people rescued";
            case FAILURE -> "Robot entered a hazard";
            case TRUNCATED -> "Step limit reached";
        });
        steps.setText(observation.steps() + " / " + maxSteps);
        reward.setText(String.format(Locale.ROOT, "%.1f", observation.totalReward()));
        rescued.setText(observation.rescued() + " / " + victimCount);
        carrying.setText(observation.carrying() ? "One person" : "Nobody");
        invalid.setText(Integer.toString(observation.invalidActions()));
        modelStatus.setText(modelEpisodes > 0
                ? "Trained · " + modelEpisodes + " episodes" : "No trained model");
    }

    void resetInspector() {
        lastAction.setText("—");
        qValues.setText("Select a neural agent to see Q-values.");
    }

    void setLastAction(String value) { lastAction.setText(value); }
    void setQValues(String value) { qValues.setText(value); }
}
