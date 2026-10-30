package it.farrlor.neurorescue;

import it.farrlor.neurorescue.simulation.Observation;
import it.farrlor.neurorescue.simulation.Position;
import it.farrlor.neurorescue.simulation.Scenario;
import it.farrlor.neurorescue.simulation.Tile;
import javafx.animation.KeyFrame;
import javafx.animation.KeyValue;
import javafx.animation.Timeline;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.geometry.Insets;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.layout.Priority;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.util.Duration;

final class SimulationView extends VBox {
    private static final Color FLOOR = Color.web("#172229");
    private static final Color WALL = Color.web("#3a4b53");
    private static final Color BORDER = Color.web("#263842");
    private static final Color TEAL = Color.web("#4fd1c5");
    private static final Color AMBER = Color.web("#f6bb65");

    private final Canvas canvas = new Canvas();
    private final DoubleProperty displayX = new SimpleDoubleProperty();
    private final DoubleProperty displayY = new SimpleDoubleProperty();
    private final Label title = new Label();
    private final Label note = new Label();
    private Scenario scenario;
    private Observation observation;
    private Timeline movement;

    SimulationView() {
        super(12);
        title.getStyleClass().add("eyebrow");
        note.getStyleClass().add("muted");
        note.setWrapText(true);
        StackPane surface = new StackPane(canvas);
        surface.getStyleClass().add("viewport");
        VBox.setVgrow(surface, Priority.ALWAYS);
        canvas.widthProperty().bind(surface.widthProperty().subtract(24));
        canvas.heightProperty().bind(surface.heightProperty().subtract(24));
        canvas.widthProperty().addListener((observable, oldValue, newValue) -> draw());
        canvas.heightProperty().addListener((observable, oldValue, newValue) -> draw());
        displayX.addListener((observable, oldValue, newValue) -> draw());
        displayY.addListener((observable, oldValue, newValue) -> draw());
        getChildren().addAll(title, surface, note);
        setPadding(new Insets(25));
        setPreview(false);
    }

    void setPreview(boolean preview) {
        title.setText(preview ? "SCENARIO PREVIEW" : "ENVIRONMENT / LIVE VIEW");
        note.setText(preview
                ? "This map is a preview. Training and evaluation episodes run separately."
                : "Static hazards end an episode. All maps are verified as safely solvable.");
    }

    void reset(Scenario scenario, Observation observation) {
        stop();
        this.scenario = scenario;
        this.observation = observation;
        displayX.set(observation.robot().x());
        displayY.set(observation.robot().y());
        draw();
    }

    void show(Scenario scenario, Observation observation) {
        this.scenario = scenario;
        this.observation = observation;
        draw();
    }

    void move(Position previous, Observation current) {
        observation = current;
        stop();
        if (previous.equals(current.robot())) {
            draw();
            return;
        }
        movement = new Timeline(new KeyFrame(Duration.millis(150),
                new KeyValue(displayX, current.robot().x()),
                new KeyValue(displayY, current.robot().y())));
        movement.play();
    }

    void stop() {
        if (movement != null) movement.stop();
        movement = null;
    }

    private void draw() {
        if (scenario == null || observation == null || canvas.getWidth() <= 0 || canvas.getHeight() <= 0) return;
        GraphicsContext graphics = canvas.getGraphicsContext2D();
        double width = canvas.getWidth();
        double height = canvas.getHeight();
        graphics.setFill(Color.web("#10191e"));
        graphics.fillRect(0, 0, width, height);
        double cell = Math.min((width - 32) / scenario.width(), (height - 32) / scenario.height());
        double left = (width - cell * scenario.width()) / 2;
        double top = (height - cell * scenario.height()) / 2;
        for (int y = 0; y < scenario.height(); y++) {
            for (int x = 0; x < scenario.width(); x++) {
                Position position = new Position(x, y);
                double px = left + x * cell;
                double py = top + y * cell;
                Tile tile = scenario.tileAt(position);
                graphics.setFill(switch (tile) {
                    case FLOOR -> FLOOR;
                    case WALL -> WALL;
                    case OBSTACLE -> Color.web("#56606b");
                    case EXIT -> Color.web("#174b47");
                    case HAZARD -> Color.web("#693933");
                });
                graphics.fillRoundRect(px + 1, py + 1, cell - 2, cell - 2, 5, 5);
                graphics.setStroke(BORDER);
                graphics.strokeRoundRect(px + 1, py + 1, cell - 2, cell - 2, 5, 5);
                if (tile == Tile.EXIT) {
                    graphics.setStroke(TEAL);
                    graphics.setLineWidth(Math.max(2, cell * 0.05));
                    graphics.strokeRoundRect(px + cell * 0.23, py + cell * 0.23,
                            cell * 0.54, cell * 0.54, cell * 0.08, cell * 0.08);
                } else if (tile == Tile.HAZARD) {
                    graphics.setFill(Color.web("#ee7b6b"));
                    graphics.fillText("!", px + cell * 0.44, py + cell * 0.67);
                }
                if (observation.victims().contains(position)) {
                    graphics.setFill(AMBER);
                    graphics.fillOval(px + cell * 0.34, py + cell * 0.29, cell * 0.32, cell * 0.32);
                    graphics.setStroke(AMBER);
                    graphics.strokeOval(px + cell * 0.25, py + cell * 0.20, cell * 0.50, cell * 0.50);
                }
            }
        }
        double rx = left + displayX.get() * cell + cell / 2;
        double ry = top + displayY.get() * cell + cell / 2;
        graphics.setFill(TEAL);
        graphics.fillOval(rx - cell * 0.29, ry - cell * 0.29, cell * 0.58, cell * 0.58);
        graphics.setFill(Color.web("#07353a"));
        graphics.fillOval(rx - cell * 0.13, ry - cell * 0.13, cell * 0.26, cell * 0.26);
        if (observation.carrying()) {
            graphics.setStroke(AMBER);
            graphics.setLineWidth(3);
            graphics.strokeOval(rx - cell * 0.35, ry - cell * 0.35, cell * 0.70, cell * 0.70);
        }
    }
}
