package it.farrlor.neurorescue;

import it.farrlor.neurorescue.ai.NeuralLab;
import it.farrlor.neurorescue.ai.TrainingScenarios;
import it.farrlor.neurorescue.simulation.Scenario;
import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;

import java.util.List;
import java.util.Locale;

final class TrainingPanel extends VBox {
    enum MapSet {
        GROUNDS("Training grounds"),
        OBSTACLES("Obstacle course"),
        HAZARDS("Hazard course"),
        CURRICULUM("Progressive curriculum");

        private final String label;

        MapSet(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }

        List<Scenario> scenarios() {
            return switch (this) {
                case GROUNDS, CURRICULUM -> TrainingScenarios.grounds();
                case OBSTACLES -> TrainingScenarios.obstacleCourse();
                case HAZARDS -> TrainingScenarios.hazardousCourse();
            };
        }
    }

    final ChoiceBox<MapSet> maps = new ChoiceBox<>();
    final TextField seed = new TextField("17");
    final TextField episodes = new TextField("600");
    final Button start = new Button("Start training");
    final Button pause = new Button("Pause");
    final Button stop = new Button("Stop");
    final Button newModel = new Button("New");
    final Button saveModel = new Button("Save");
    final Button loadModel = new Button("Load");

    private final Label status = new Label();
    private final Label metrics = new Label("No episodes yet");
    private final Label operation = new Label();
    private final TrainingChart rewardChart = new TrainingChart(
            "Episode reward", "Raw environment reward · 20-episode mean", Color.web("#4fd1c5"), true);
    private final TrainingChart successChart = new TrainingChart(
            "Evaluation success", "Greedy measurements on fixed maps", Color.web("#85c89c"), false);
    private final TrainingChart lossChart = new TrainingChart(
            "Optimization loss", "Measured Huber loss; warmup is omitted", Color.web("#f6bb65"), false);
    private TrainingStatus trainingStatus = TrainingStatus.IDLE;
    private int completedEpisodes;
    private int requestedEpisodes;
    private double lastEvaluationSuccess = Double.NaN;

    TrainingPanel() {
        super(11);
        getStyleClass().add("side-panel");
        Label title = new Label("TRAIN THE RESCUE AGENT");
        title.getStyleClass().add("panel-title");
        Label hint = new Label("Episodes run headlessly. The map is a preview, not the current training episode.");
        hint.getStyleClass().add("muted");
        hint.setWrapText(true);
        maps.getItems().addAll(MapSet.values());
        maps.setValue(MapSet.GROUNDS);
        maps.getStyleClass().add("selector");
        seed.setTooltip(new Tooltip("Seed for exploration and replay sampling"));
        episodes.setTooltip(new Tooltip("Number of complete attempts. Each episode resets the map."));
        newModel.setTooltip(new Tooltip("Discard the current network and create fresh random weights"));
        saveModel.setTooltip(new Tooltip("Save inference weights and metadata; replay and optimizer are not saved"));
        loadModel.setTooltip(new Tooltip("Load an inference model from a previous session"));
        status.setWrapText(true);
        status.getStyleClass().add("training-status");
        metrics.setWrapText(true);
        metrics.getStyleClass().add("training-metrics");
        operation.setWrapText(true);
        operation.getStyleClass().add("muted");
        HBox runControls = new HBox(7, start, pause, stop);
        HBox modelControls = new HBox(7, newModel, saveModel, loadModel);
        runControls.setAlignment(Pos.CENTER_LEFT);
        Label modelTitle = new Label("MODEL");
        modelTitle.getStyleClass().add("eyebrow");
        getChildren().addAll(title, hint, labeled("TRAINING MAPS", maps),
                labeled("EPISODES", episodes), labeled("TRAINING SEED", seed),
                runControls, modelTitle, modelControls, labeled("TRAINING STATUS", status),
                metrics, operation, rewardChart, successChart, lossChart);
        setStatus(TrainingStatus.IDLE, 0, 0);
    }

    private static VBox labeled(String title, javafx.scene.Node control) {
        Label label = new Label(title);
        label.getStyleClass().add("eyebrow");
        return new VBox(5, label, control);
    }

    void setStatus(TrainingStatus state, int completed, int requested) {
        trainingStatus = state;
        completedEpisodes = completed;
        requestedEpisodes = requested;
        status.setText(state.label(completed, requested));
        pause.setText(state == TrainingStatus.PAUSED ? "Resume" : "Pause");
    }

    TrainingStatus status() {
        return trainingStatus;
    }

    int completedEpisodes() {
        return completedEpisodes;
    }

    int requestedEpisodes() {
        return requestedEpisodes;
    }

    void showSnapshot(NeuralLab.TrainingSnapshot snapshot) {
        var progress = snapshot.progress();
        completedEpisodes = progress.completedEpisodes();
        requestedEpisodes = progress.requestedEpisodes();
        var episode = progress.episode();
        rewardChart.add(progress.completedEpisodes(), episode.reward(), progress.rollingReward());
        lossChart.add(progress.completedEpisodes(), episode.meanLoss(), Double.NaN);
        if (snapshot.evaluation() != null) {
            lastEvaluationSuccess = snapshot.evaluation().successRate();
            successChart.add(progress.completedEpisodes(), lastEvaluationSuccess * 100, Double.NaN);
        }
        metrics.setText(String.format(Locale.ROOT,
                "Episode %d / %d · ε %.3f%nReward mean (20): %.1f%nTraining success (20): %.0f%%%nLoss: %s%nReplay: %d transitions%nThroughput: %.0f steps/s%nLast greedy evaluation: %s",
                progress.completedEpisodes(), progress.requestedEpisodes(), progress.epsilon(),
                progress.rollingReward(), progress.rollingTrainingSuccess() * 100,
                Double.isNaN(episode.meanLoss()) ? "warming up"
                        : String.format(Locale.ROOT, "%.3f", episode.meanLoss()),
                progress.replaySize(), progress.transitionsPerSecond(),
                Double.isNaN(lastEvaluationSuccess) ? "unavailable"
                        : String.format(Locale.ROOT, "%.0f%%", lastEvaluationSuccess * 100)));
        status.setText(trainingStatus.label(completedEpisodes, requestedEpisodes));
    }

    void clearHistory(String message) {
        rewardChart.clear();
        successChart.clear();
        lossChart.clear();
        lastEvaluationSuccess = Double.NaN;
        metrics.setText(message);
    }

    void setOperationMessage(String message) {
        operation.setText(message);
    }

    void updateControls(boolean busy, boolean modelOperationPending, boolean modelAvailable) {
        start.setDisable(busy || modelOperationPending);
        pause.setDisable(!busy);
        stop.setDisable(!busy);
        newModel.setDisable(busy || modelOperationPending);
        saveModel.setDisable(busy || modelOperationPending || !modelAvailable);
        loadModel.setDisable(busy || modelOperationPending);
    }
}
