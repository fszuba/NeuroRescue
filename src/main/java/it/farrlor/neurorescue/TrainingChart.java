package it.farrlor.neurorescue;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;

import java.util.List;
import java.util.Locale;

/** A bounded chart of measurements at their actual episode indices. */
final class TrainingChart extends VBox {
    private final TrainingChartData data = new TrainingChartData(100);
    private final Canvas canvas = new Canvas(265, 92);
    private final Color color;
    private final boolean connectAverages;

    TrainingChart(String title, String explanation, Color color, boolean connectAverages) {
        super(4);
        this.color = color;
        this.connectAverages = connectAverages;
        Label heading = new Label(title);
        heading.getStyleClass().add("chart-title");
        Label note = new Label(explanation);
        note.getStyleClass().add("muted");
        note.setWrapText(true);
        getChildren().addAll(heading, canvas, note);
        getStyleClass().add("chart-card");
        canvas.widthProperty().addListener((observable, oldValue, newValue) -> draw());
        widthProperty().addListener((observable, oldValue, newValue) ->
                canvas.setWidth(Math.max(160, newValue.doubleValue() - 26)));
        draw();
    }

    void add(int episode, double value, double average) {
        data.add(episode, value, average);
        draw();
    }

    void clear() {
        data.clear();
        draw();
    }

    private void draw() {
        GraphicsContext graphics = canvas.getGraphicsContext2D();
        double width = canvas.getWidth();
        double height = canvas.getHeight();
        graphics.setFill(Color.web("#18252b"));
        graphics.fillRect(0, 0, width, height);
        List<TrainingChartData.Sample> samples = data.samples();
        if (samples.isEmpty()) return;

        double minimum = samples.stream().mapToDouble(TrainingChartData.Sample::value).min().orElse(0);
        double maximum = samples.stream().mapToDouble(TrainingChartData.Sample::value).max().orElse(1);
        for (TrainingChartData.Sample sample : samples) {
            if (Double.isFinite(sample.average())) {
                minimum = Math.min(minimum, sample.average());
                maximum = Math.max(maximum, sample.average());
            }
        }
        if (maximum - minimum < 0.01) {
            maximum += 0.5;
            minimum -= 0.5;
        }
        double left = 8;
        double right = width - 8;
        double top = 8;
        double bottom = height - 21;
        graphics.setStroke(Color.web("#30424b"));
        graphics.strokeLine(left, bottom, right, bottom);
        graphics.setFill(Color.web("#9bb1b8"));
        graphics.setFont(Font.font(10));
        graphics.fillText("ep " + samples.getFirst().episode(), left, height - 5);
        String end = "ep " + samples.getLast().episode();
        graphics.fillText(end, right - 6 * end.length(), height - 5);
        graphics.fillText(String.format(Locale.ROOT, "%.1f", maximum), left, top + 9);
        graphics.fillText(String.format(Locale.ROOT, "%.1f", minimum), left, bottom - 3);

        double previousX = Double.NaN;
        double previousY = Double.NaN;
        for (TrainingChartData.Sample sample : samples) {
            double x = left + data.horizontalPosition(sample.episode()) * (right - left);
            double y = bottom - (sample.value() - minimum) / (maximum - minimum) * (bottom - top);
            graphics.setFill(color.deriveColor(0, 1, 1, 0.65));
            graphics.fillOval(x - 2, y - 2, 4, 4);
            if (Double.isFinite(sample.average())) {
                double averageY = bottom - (sample.average() - minimum) / (maximum - minimum) * (bottom - top);
                if (connectAverages && Double.isFinite(previousX)) {
                    graphics.setStroke(color);
                    graphics.setLineWidth(1.7);
                    graphics.strokeLine(previousX, previousY, x, averageY);
                }
                graphics.setFill(color);
                graphics.fillOval(x - 2.5, averageY - 2.5, 5, 5);
                previousX = x;
                previousY = averageY;
            } else {
                previousX = Double.NaN;
            }
        }
    }
}
