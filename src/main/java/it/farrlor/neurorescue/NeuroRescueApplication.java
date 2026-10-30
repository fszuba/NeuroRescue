package it.farrlor.neurorescue;

import it.farrlor.neurorescue.baseline.RouteAgent;
import it.farrlor.neurorescue.ai.DqnAgent;
import it.farrlor.neurorescue.ai.Evaluation;
import it.farrlor.neurorescue.ai.NeuralLab;
import it.farrlor.neurorescue.ai.TrainingScenarios;
import it.farrlor.neurorescue.simulation.Action;
import it.farrlor.neurorescue.simulation.EpisodeStatus;
import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Position;
import it.farrlor.neurorescue.simulation.RescueEnvironment;
import it.farrlor.neurorescue.simulation.RewardScheme;
import it.farrlor.neurorescue.simulation.Scenario;
import it.farrlor.neurorescue.simulation.Scenarios;
import it.farrlor.neurorescue.simulation.StepResult;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Alert;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextField;
import javafx.scene.control.TextInputControl;
import javafx.scene.control.Tooltip;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletionException;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Presentation layer for the simulation, background training, and fixed-seed evaluation. */
public final class NeuroRescueApplication extends Application {
    private static final Logger LOG = Logger.getLogger(NeuroRescueApplication.class.getName());
    private enum Mode {
        SIMULATION("Simulation"), TRAINING("Training"), EVALUATION("Evaluation");
        private final String label;

        Mode(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }
    }

    private enum Agent {
        ROUTE("Shortest safe route"), MANUAL("Manual controls"),
        UNTRAINED("Untrained neural"), TRAINED("Trained neural");
        private final String label;

        Agent(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }

        boolean neural() {
            return this == UNTRAINED || this == TRAINED;
        }

        NeuralLab.EvaluationAgent evaluationAgent() {
            return switch (this) {
                case ROUTE -> NeuralLab.EvaluationAgent.ROUTE;
                case UNTRAINED -> NeuralLab.EvaluationAgent.UNTRAINED;
                case TRAINED -> NeuralLab.EvaluationAgent.TRAINED;
                case MANUAL -> throw new IllegalStateException("Manual control cannot be evaluated");
            };
        }
    }

    private enum EvaluationMap {
        GROUNDS("Training grounds"), HELD_OUT("Held-out grounds"),
        OBSTACLES("Obstacle course"), HAZARDS("Hazard course"), BUILT_INS("Built-in maps");
        private final String label;

        EvaluationMap(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return label;
        }

        List<Scenario> scenarios() {
            return switch (this) {
                case GROUNDS -> TrainingScenarios.grounds();
                case HELD_OUT -> TrainingScenarios.heldOutGrounds();
                case OBSTACLES -> TrainingScenarios.obstacleCourse();
                case HAZARDS -> TrainingScenarios.hazardousCourse();
                case BUILT_INS -> Scenarios.builtIns();
            };
        }
    }

    private final RouteAgent baseline = new RouteAgent();
    private final NeuralLab neuralLab = new NeuralLab();
    private final SimulationView simulationView = new SimulationView();
    private final MissionPanel missionPanel = new MissionPanel();
    private final TrainingPanel trainingPanel = new TrainingPanel();
    private final Label evaluationResults = new Label("Run evaluation to see results.");
    private final Label footerHelp = new Label();
    private final ChoiceBox<Mode> modes = new ChoiceBox<>();
    private final ChoiceBox<Scenario> scenarios = new ChoiceBox<>();
    private final ChoiceBox<Agent> agents = new ChoiceBox<>();
    private final ChoiceBox<EvaluationMap> evaluationMaps = new ChoiceBox<>();
    private final TextField evaluationEpisodesInput = new TextField("30");
    private final Button play = new Button("▶  Run");
    private final Button step = new Button("Step");
    private final Button resetButton = new Button("Reset");
    private final Button evaluateButton = new Button("Run evaluation");
    private final HBox simulationActions = new HBox(7);
    private VBox agentChoice;
    private final StackPane inspectorContent = new StackPane();
    private Timeline playback;
    private RescueEnvironment environment;
    private Scenario generatedScenario;
    private int simulationGeneration;
    private boolean inferencePending;
    private int modelEpisodes;
    private boolean modelOperationPending;

    @Override
    public void start(Stage stage) {
        List<Scenario> builtIns = Scenarios.builtIns();
        scenarios.getItems().addAll(TrainingScenarios.grounds());
        scenarios.getItems().addAll(TrainingScenarios.obstacleCourse());
        scenarios.getItems().addAll(TrainingScenarios.hazardousCourse());
        scenarios.getItems().addAll(builtIns);
        scenarios.setConverter(new javafx.util.StringConverter<>() {
            @Override public String toString(Scenario scenario) { return scenario == null ? "" : scenario.name(); }
            @Override public Scenario fromString(String value) { throw new UnsupportedOperationException(); }
        });
        scenarios.setValue(TrainingScenarios.grounds().getFirst());
        agents.getItems().addAll(Agent.values());
        agents.setValue(Agent.ROUTE);
        modes.getItems().addAll(Mode.values());
        modes.setValue(Mode.SIMULATION);
        evaluationMaps.getItems().addAll(EvaluationMap.values());
        evaluationMaps.setValue(EvaluationMap.GROUNDS);

        BorderPane root = new BorderPane();
        root.getStyleClass().add("app");
        root.setTop(toolbar());
        root.setCenter(simulationView);
        root.setRight(inspector());
        root.setBottom(footer());

        Scene scene = new Scene(root, 1180, 750);
        scene.getStylesheets().add(getClass().getResource("/it/farrlor/neurorescue/app.css").toExternalForm());
        scene.setOnKeyPressed(event -> {
            if (scene.getFocusOwner() instanceof TextInputControl || modes.getValue() != Mode.SIMULATION) return;
            if (event.getCode() == KeyCode.SPACE) {
                togglePlayback();
                event.consume();
            } else if (agents.getValue() == Agent.MANUAL) {
                Action action = switch (event.getCode()) {
                    case UP, W -> Action.UP;
                    case DOWN, S -> Action.DOWN;
                    case LEFT, A -> Action.LEFT;
                    case RIGHT, D -> Action.RIGHT;
                    case P -> Action.PICK_UP;
                    case E -> Action.DROP_OFF;
                    default -> null;
                };
                if (action != null) {
                    advance(action);
                    event.consume();
                }
            }
        });
        scenarios.setOnAction(event -> reset());
        agents.setOnAction(event -> {
            stopPlayback();
            simulationGeneration++;
            inferencePending = false;
            missionPanel.setQValues("—");
            refresh();
        });
        modes.setOnAction(event -> switchMode());
        missionPanel.loadModel.setOnAction(event -> loadModel());
        missionPanel.generate.setOnAction(event -> generateMap());
        missionPanel.seed.setOnAction(event -> generateMap());
        trainingPanel.start.setOnAction(event -> startTraining());
        trainingPanel.pause.setOnAction(event -> toggleTrainingPause());
        trainingPanel.stop.setOnAction(event -> neuralLab.stopTraining());
        trainingPanel.newModel.setOnAction(event -> createModel());
        trainingPanel.saveModel.setOnAction(event -> saveModel());
        trainingPanel.loadModel.setOnAction(event -> loadModel());
        missionPanel.speed.valueProperty().addListener((observable, oldValue, newValue) -> {
            if (playback != null) playback.setRate(newValue.doubleValue());
        });
        play.setOnAction(event -> togglePlayback());
        step.setOnAction(event -> advanceAgent());
        resetButton.setOnAction(event -> reset());
        reset();
        switchMode();
        updateTrainingControls();
        stage.setTitle("NeuroRescue · Simulation Lab");
        stage.setScene(scene);
        stage.setMinWidth(960);
        stage.setMinHeight(620);
        stage.show();
    }

    private HBox toolbar() {
        Label brand = new Label("NEURORESCUE");
        brand.getStyleClass().add("brand");
        Label subtitle = new Label("RESCUE LEARNING LAB");
        subtitle.getStyleClass().add("eyebrow");
        VBox identity = new VBox(3, brand, subtitle);
        HBox.setHgrow(identity, Priority.ALWAYS);
        simulationActions.getChildren().addAll(play, step, resetButton);
        agentChoice = labeledChoice("AGENT", agents);
        HBox bar = new HBox(12, identity, labeledChoice("MODE", modes),
                labeledChoice("SCENARIO", scenarios), agentChoice, simulationActions);
        bar.setAlignment(Pos.CENTER_LEFT);
        bar.getStyleClass().add("toolbar");
        return bar;
    }

    private VBox labeledChoice(String title, ChoiceBox<?> choice) {
        Label label = new Label(title);
        label.getStyleClass().add("eyebrow");
        choice.getStyleClass().add("selector");
        return new VBox(5, label, choice);
    }

    private ScrollPane inspector() {
        inspectorContent.getChildren().addAll(missionPanel, trainingPanel, evaluationPanel());
        ScrollPane scroll = new ScrollPane(inspectorContent);
        scroll.setFitToWidth(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.getStyleClass().add("inspector-scroll");
        scroll.setPrefWidth(350);
        return scroll;
    }

    private VBox evaluationPanel() {
        Label title = new Label("EVALUATE POLICY");
        title.getStyleClass().add("panel-title");
        Label hint = new Label("Greedy neural inference uses no exploration or training updates. Scenarios repeat across fixed evaluation episodes.");
        hint.getStyleClass().add("muted");
        hint.setWrapText(true);
        evaluationEpisodesInput.setTooltip(new Tooltip("Episodes rotate through the selected map set with a fixed seed"));
        evaluateButton.setOnAction(event -> evaluate());
        evaluationResults.setWrapText(true);
        VBox panel = new VBox(16, title, hint, labeledChoice("MAP SET", evaluationMaps),
                labeledInput("EPISODES", evaluationEpisodesInput), evaluateButton,
                metric("RESULTS", evaluationResults));
        panel.getStyleClass().add("side-panel");
        return panel;
    }

    private VBox labeledInput(String title, TextField field) {
        Label label = new Label(title);
        label.getStyleClass().add("eyebrow");
        return new VBox(5, label, field);
    }

    private VBox metric(String name, Label value) {
        Label label = new Label(name);
        label.getStyleClass().add("eyebrow");
        value.getStyleClass().add("metric-value");
        return new VBox(3, label, value);
    }

    private HBox footer() {
        Label label = new Label("LOCAL SIMULATION  ·  REPRODUCIBLE SCENARIOS");
        label.getStyleClass().add("eyebrow");
        footerHelp.getStyleClass().add("muted");
        Region spacer = new Region();
        HBox.setHgrow(spacer, Priority.ALWAYS);
        HBox footer = new HBox(label, spacer, footerHelp);
        footer.getStyleClass().add("footer");
        return footer;
    }

    private void reset() {
        stopPlayback();
        simulationGeneration++;
        inferencePending = false;
        environment = new RescueEnvironment(scenarios.getValue(), RewardScheme.defaults(), 200);
        Observation observation = environment.reset(0);
        simulationView.reset(environment.scenario(), observation);
        missionPanel.resetInspector();
        refresh();
    }

    private void generateMap() {
        try {
            long seed = Long.parseLong(missionPanel.seed.getText().trim());
            Scenario scenario = Scenarios.generated(seed);
            if (generatedScenario != null) scenarios.getItems().remove(generatedScenario);
            generatedScenario = scenario;
            scenarios.getItems().add(scenario);
            scenarios.setValue(scenario);
        } catch (NumberFormatException invalidSeed) {
            Alert alert = new Alert(Alert.AlertType.ERROR, "Enter a whole-number seed between -9223372036854775808 and 9223372036854775807.");
            alert.setHeaderText("Invalid map seed");
            alert.showAndWait();
        }
    }

    private void togglePlayback() {
        if (environment.observation().status() != EpisodeStatus.RUNNING
                || agents.getValue() == Agent.MANUAL || modes.getValue() != Mode.SIMULATION) return;
        if (playback != null) {
            stopPlayback();
        } else {
            playback = new Timeline(new KeyFrame(Duration.millis(240), event -> advanceAgent()));
            playback.setCycleCount(Timeline.INDEFINITE);
            playback.setRate(missionPanel.speed.getValue());
            playback.play();
            play.setText("Ⅱ  Pause");
        }
    }

    private void stopPlayback() {
        if (playback != null) playback.stop();
        playback = null;
        play.setText("▶  Run");
    }

    private void advanceAgent() {
        if (environment.observation().status() != EpisodeStatus.RUNNING || inferencePending) return;
        Agent selected = agents.getValue();
        if (selected == Agent.ROUTE) {
            advance(baseline.choose(environment.scenario(), environment.observation()));
        } else if (selected.neural()) {
            if (neuralLab.training() || modelOperationPending
                    || (selected == Agent.TRAINED && modelEpisodes == 0)) return;
            inferencePending = true;
            int generation = simulationGeneration;
            int currentStep = environment.observation().steps();
            neuralLab.infer(selected == Agent.UNTRAINED, environment.scenario(), environment.observation())
                    .whenComplete((decision, error) -> Platform.runLater(() -> {
                        if (generation != simulationGeneration || currentStep != environment.observation().steps()) return;
                        inferencePending = false;
                        if (error != null) { stopPlayback(); showError("Neural inference failed", error); return; }
                        missionPanel.setQValues(formatQValues(decision));
                        advance(decision.action());
                    }));
        }
    }

    private void advance(Action action) {
        if (environment.observation().status() != EpisodeStatus.RUNNING) return;
        Position previous = environment.observation().robot();
        StepResult result = environment.step(action);
        missionPanel.setLastAction(action.name().replace('_', ' ')
                + (result.valid() ? "" : " · invalid"));
        simulationView.move(previous, result.observation());
        if (result.observation().status() != EpisodeStatus.RUNNING) stopPlayback();
        refresh();
    }

    private void refresh() {
        if (environment == null) return;
        Observation observation = environment.observation();
        missionPanel.show(observation, environment.maxSteps(),
                environment.scenario().victims().size(), modelEpisodes);
        boolean automatic = agents.getValue() != Agent.MANUAL;
        boolean unavailableModel = agents.getValue() == Agent.TRAINED && modelEpisodes == 0;
        boolean busy = (neuralLab.training() || modelOperationPending) && agents.getValue().neural();
        play.setDisable(!automatic || unavailableModel || busy || observation.status() != EpisodeStatus.RUNNING);
        step.setDisable(!automatic || unavailableModel || busy || observation.status() != EpisodeStatus.RUNNING);
        simulationView.show(environment.scenario(), observation);
    }

    private void switchMode() {
        stopPlayback();
        simulationGeneration++;
        inferencePending = false;
        Mode mode = modes.getValue();
        for (int i = 0; i < inspectorContent.getChildren().size(); i++) {
            boolean active = switch (i) {
                case 0 -> mode == Mode.SIMULATION;
                case 1 -> mode == Mode.TRAINING;
                default -> mode == Mode.EVALUATION;
            };
            inspectorContent.getChildren().get(i).setVisible(active);
            inspectorContent.getChildren().get(i).setManaged(active);
        }
        boolean simulation = mode == Mode.SIMULATION;
        simulationActions.setVisible(simulation);
        simulationActions.setManaged(simulation);
        agentChoice.setVisible(mode != Mode.TRAINING);
        agentChoice.setManaged(mode != Mode.TRAINING);
        simulationView.setPreview(!simulation);
        footerHelp.setText(switch (mode) {
            case SIMULATION -> "Space: run / pause   •   Manual agent: arrow keys or WASD";
            case TRAINING -> "Training runs on a background CPU worker";
            case EVALUATION -> "Evaluation uses fixed episodes without training updates";
        });
        if (mode == Mode.EVALUATION && agents.getValue() == Agent.MANUAL) {
            agents.setValue(Agent.ROUTE);
        }
        refresh();
    }

    private void startTraining() {
        if (neuralLab.training() || modelOperationPending) return;
        try {
            int episodes = positiveInt(trainingPanel.episodes, "episode count", 20_000);
            long seed = Long.parseLong(trainingPanel.seed.getText().trim());
            boolean curriculum = trainingPanel.maps.getValue() == TrainingPanel.MapSet.CURRICULUM;
            List<Scenario> maps = trainingPanel.maps.getValue().scenarios();
            scenarios.setValue(maps.getFirst());
            int target = modelEpisodes + episodes;
            trainingPanel.setStatus(TrainingStatus.STARTING, modelEpisodes, target);
            trainingPanel.setOperationMessage("");
            updateTrainingControls();
            neuralLab.train(maps, episodes, seed, curriculum,
                    snapshot -> Platform.runLater(() -> showTrainingSnapshot(snapshot)))
                    .whenComplete((result, error) -> Platform.runLater(() -> {
                        if (error != null) {
                            trainingPanel.setStatus(TrainingStatus.FAILED,
                                    trainingPanel.completedEpisodes(), target);
                            showError("Training failed", error);
                        } else {
                            modelEpisodes = result.completedEpisodes();
                            TrainingStatus status = result.outcome() == NeuralLab.TrainingOutcome.COMPLETED
                                    ? TrainingStatus.COMPLETED : TrainingStatus.STOPPED;
                            trainingPanel.setStatus(status, modelEpisodes, result.requestedEpisodes());
                        }
                        updateTrainingControls();
                        refresh();
                    }));
            updateTrainingControls();
            refresh();
        } catch (NumberFormatException invalid) {
            showError("Invalid training settings", invalid);
        }
    }

    private void showTrainingSnapshot(NeuralLab.TrainingSnapshot snapshot) {
        var progress = snapshot.progress();
        modelEpisodes = progress.completedEpisodes();
        trainingPanel.showSnapshot(snapshot);
        if (trainingPanel.status() == TrainingStatus.STARTING) {
            trainingPanel.setStatus(TrainingStatus.TRAINING,
                    progress.completedEpisodes(), progress.requestedEpisodes());
        }
    }

    private void toggleTrainingPause() {
        if (!neuralLab.training()) return;
        if (neuralLab.paused()) neuralLab.resume(); else neuralLab.pause();
        trainingPanel.setStatus(neuralLab.paused() ? TrainingStatus.PAUSED : TrainingStatus.TRAINING,
                trainingPanel.completedEpisodes(), trainingPanel.requestedEpisodes());
    }

    private void updateTrainingControls() {
        boolean busy = neuralLab.training();
        trainingPanel.updateControls(busy, modelOperationPending, modelEpisodes > 0);
        missionPanel.loadModel.setDisable(busy || modelOperationPending);
        evaluateButton.setDisable(busy || modelOperationPending);
    }

    private void createModel() {
        if (neuralLab.training() || modelOperationPending) return;
        try {
            long seed = Long.parseLong(trainingPanel.seed.getText().trim());
            modelOperationPending = true;
            trainingPanel.setOperationMessage("Initializing new model…");
            updateTrainingControls(); refresh();
            neuralLab.newModel(seed).whenComplete((ignored, error) -> Platform.runLater(() -> {
                modelOperationPending = false;
                if (error != null) {
                    trainingPanel.setOperationMessage("Model initialization failed");
                    updateTrainingControls(); refresh();
                    showError("Could not create model", error);
                    return;
                }
                modelEpisodes = 0;
                trainingPanel.clearHistory("New untrained model · ready to train");
                trainingPanel.setStatus(TrainingStatus.IDLE, 0, 0);
                trainingPanel.setOperationMessage("New model ready");
                updateTrainingControls(); refresh();
            }));
        } catch (NumberFormatException invalid) { showError("Invalid training seed", invalid); }
    }

    private void saveModel() {
        if (neuralLab.training() || modelOperationPending || modelEpisodes == 0) return;
        FileChooser chooser = modelChooser();
        chooser.setInitialFileName("neurorescue-" + modelEpisodes + ".nrdq");
        java.io.File chosen = chooser.showSaveDialog(missionPanel.getScene().getWindow());
        if (chosen == null) return;
        modelOperationPending = true;
        trainingPanel.setOperationMessage("Saving inference model…");
        updateTrainingControls(); refresh();
        neuralLab.save(chosen.toPath()).whenComplete((ignored, error) -> Platform.runLater(() -> {
            modelOperationPending = false;
            if (error != null) {
                trainingPanel.setOperationMessage("Save failed");
                showError("Could not save model", error);
            } else {
                trainingPanel.setOperationMessage("Saved inference model · " + modelEpisodes + " episodes");
            }
            updateTrainingControls(); refresh();
        }));
    }

    private void loadModel() {
        if (neuralLab.training() || modelOperationPending) return;
        java.io.File chosen = modelChooser().showOpenDialog(missionPanel.getScene().getWindow());
        if (chosen == null) return;
        modelOperationPending = true;
        trainingPanel.setOperationMessage("Loading model…");
        updateTrainingControls(); refresh();
        neuralLab.load(chosen.toPath()).whenComplete((state, error) -> Platform.runLater(() -> {
            modelOperationPending = false;
            if (error != null) {
                trainingPanel.setOperationMessage("Load failed");
                updateTrainingControls(); refresh();
                showError("Could not load model", error);
                return;
            }
            modelEpisodes = state.completedEpisodes();
            trainingPanel.setStatus(TrainingStatus.IDLE, modelEpisodes, modelEpisodes);
            trainingPanel.clearHistory("Loaded weights · training history is not stored in this file");
            trainingPanel.setOperationMessage("Loaded inference model · " + modelEpisodes + " episodes");
            updateTrainingControls(); refresh();
        }));
    }

    private FileChooser modelChooser() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("NeuroRescue inference model");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("NeuroRescue model (*.nrdq)", "*.nrdq"));
        return chooser;
    }

    private void evaluate() {
        if (neuralLab.training() || modelOperationPending) return;
        try {
            int episodes = positiveInt(evaluationEpisodesInput, "evaluation episode count", 1_000);
            Agent agent = agents.getValue();
            if (agent == Agent.MANUAL) throw new NumberFormatException("Select an automatic agent for evaluation");
            List<Scenario> maps = evaluationMaps.getValue().scenarios();
            evaluationResults.setText("Evaluating " + agent + "…");
            evaluateButton.setDisable(true);
            neuralLab.evaluate(agent.evaluationAgent(), maps, episodes, 2026, 64)
                    .whenComplete((result, error) -> Platform.runLater(() -> {
                evaluateButton.setDisable(false);
                if (error != null) { evaluationResults.setText("Evaluation failed"); showError("Evaluation failed", error); return; }
                evaluationResults.setText(String.format(Locale.ROOT,
                        "%d / %d rescued (%.0f%%)%nAverage reward: %.1f%nAverage steps: %.1f%nInvalid actions: %.1f / episode%nHazard encounters: %.1f / episode",
                        result.successes(), result.episodes(), result.successRate() * 100,
                        result.averageReward(), result.averageSteps(), result.averageInvalidActions(), result.averageHazards()));
                    }));
        } catch (NumberFormatException invalid) { showError("Invalid evaluation settings", invalid); }
    }

    private static int positiveInt(TextField field, String name, int maximum) {
        int value = Integer.parseInt(field.getText().trim());
        if (value < 1 || value > maximum) throw new NumberFormatException(name + " must be 1–" + maximum);
        return value;
    }

    private static String formatQValues(DqnAgent.Decision decision) {
        float[] values = decision.qValues();
        StringBuilder text = new StringBuilder();
        for (Action action : Action.values()) {
            text.append(action.name().replace('_', ' ')).append(": ")
                    .append(String.format(Locale.ROOT, "%.2f", values[action.ordinal()])).append('\n');
        }
        text.append("Selected: ").append(decision.action().name().replace('_', ' '))
                .append(decision.exploring() ? " · exploration" : " · greedy");
        return text.toString();
    }

    private void showError(String title, Throwable error) {
        Throwable cause = error instanceof CompletionException && error.getCause() != null ? error.getCause() : error;
        LOG.log(Level.SEVERE, title, cause);
        Alert alert = new Alert(Alert.AlertType.ERROR, cause.getMessage() == null ? cause.toString() : cause.getMessage());
        alert.setHeaderText(title);
        alert.show();
    }

    @Override
    public void stop() {
        stopPlayback();
        simulationView.stop();
        neuralLab.close();
    }
}
