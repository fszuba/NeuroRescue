package it.farrlor.neurorescue.ai;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DjlTrainingSmokeTest {
    @Test
    void cpuBackendProducesGradientsAndImprovesARegressionTarget() {
        try (Model model = Model.newInstance("training-smoke", "PyTorch")) {
            model.setBlock(new SequentialBlock()
                    .add(Linear.builder().setUnits(16).build())
                    .add(Activation::relu)
                    .add(Linear.builder().setUnits(2).build()));
            DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Adam.builder().optLearningRateTracker(Tracker.fixed(0.01f)).build())
                    .optDevices(new Device[]{Device.cpu()});
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 2));
                float initialLoss = Float.NaN;
                float finalLoss = Float.NaN;
                for (int iteration = 0; iteration < 120; iteration++) {
                    try (NDManager batch = trainer.getManager().newSubManager()) {
                        NDArray input = batch.create(new float[]{1f, -1f}, new Shape(1, 2));
                        NDArray expected = batch.create(new float[]{0.8f, -0.4f}, new Shape(1, 2));
                        try (GradientCollector gradients = trainer.newGradientCollector()) {
                            NDArray output = trainer.forward(new NDList(input)).singletonOrThrow();
                            assertEquals(new Shape(1, 2), output.getShape());
                            NDArray loss = output.sub(expected).square().mean();
                            if (iteration == 0) initialLoss = loss.getFloat();
                            finalLoss = loss.getFloat();
                            gradients.backward(loss);
                        }
                        trainer.step();
                    }
                }
                assertTrue(Float.isFinite(finalLoss));
                assertTrue(finalLoss < initialLoss * 0.1f,
                        "CPU optimization did not improve the target: " + initialLoss + " -> " + finalLoss);
            }
        }
    }
}
