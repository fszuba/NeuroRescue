package it.farrlor.neurorescue.ai;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import it.farrlor.neurorescue.simulation.Action;

import java.util.List;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;

/** CPU DQN with a frozen target model. Confine instances to one thread. */
public final class DqnNetwork implements AutoCloseable {
    private static final int FILE_MAGIC = 0x4E524451; // NRDQ
    private static final int FILE_VERSION = 2;
    private static final String ACTION_ORDER = java.util.Arrays.stream(Action.values())
            .map(Enum::name).collect(java.util.stream.Collectors.joining(","));
    private final DqnConfig config;
    private final Model online;
    private final Model target;
    private final Trainer trainer;
    private final Trainer targetTrainer;
    private int optimizationSteps;

    public DqnNetwork(DqnConfig config, long seed) {
        this.config = config;
        Engine.getInstance().setRandomSeed((int) (seed ^ (seed >>> 32)));
        online = Model.newInstance("neurorescue-q", "PyTorch");
        target = Model.newInstance("neurorescue-target", "PyTorch");
        online.setBlock(block(config.hiddenUnits()));
        target.setBlock(block(config.hiddenUnits()));
        DefaultTrainingConfig training = new DefaultTrainingConfig(Loss.l2Loss())
                .optOptimizer(Adam.builder()
                        .optLearningRateTracker(Tracker.fixed(config.learningRate()))
                        .build())
                .optDevices(new Device[]{Device.cpu()});
        trainer = online.newTrainer(training);
        targetTrainer = target.newTrainer(training);
        trainer.initialize(new Shape(1, ObservationEncoder.SIZE));
        targetTrainer.initialize(new Shape(1, ObservationEncoder.SIZE));
        synchronizeTarget();
    }

    private static SequentialBlock block(int hiddenUnits) {
        return new SequentialBlock()
                .add(Linear.builder().setUnits(hiddenUnits).build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(hiddenUnits).build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(Action.values().length).build());
    }

    public float[] values(float[] state) {
        return predict(trainer, state);
    }

    float[] targetValues(float[] state) {
        return predict(targetTrainer, state);
    }

    private static float[] predict(Trainer source, float[] state) {
        checkState(state);
        try (NDManager manager = source.getManager().newSubManager()) {
            NDArray input = manager.create(state, new Shape(1, ObservationEncoder.SIZE));
            float[] values = source.evaluate(new NDList(input)).singletonOrThrow().toFloatArray();
            for (float value : values) {
                if (!Float.isFinite(value)) throw new IllegalStateException("Neural inference produced a non-finite Q-value");
            }
            return values;
        }
    }

    public float trainBatch(List<Experience> batch) {
        if (batch == null || batch.isEmpty()) throw new IllegalArgumentException("DQN batch cannot be empty");
        int size = batch.size();
        int actions = Action.values().length;
        float[] states = new float[size * ObservationEncoder.SIZE];
        float[] nextStates = new float[states.length];
        float[] oneHotActions = new float[size * actions];
        for (int i = 0; i < size; i++) {
            Experience item = batch.get(i);
            System.arraycopy(item.state(), 0, states, i * ObservationEncoder.SIZE, ObservationEncoder.SIZE);
            System.arraycopy(item.nextState(), 0, nextStates, i * ObservationEncoder.SIZE, ObservationEncoder.SIZE);
            oneHotActions[i * actions + item.action()] = 1;
        }

        float[] targets = new float[size];
        try (NDManager manager = targetTrainer.getManager().newSubManager()) {
            NDArray next = manager.create(nextStates, new Shape(size, ObservationEncoder.SIZE));
            float[] nextValues = targetTrainer.evaluate(new NDList(next)).singletonOrThrow().toFloatArray();
            for (int i = 0; i < size; i++) {
                Experience item = batch.get(i);
                targets[i] = bellmanTarget(item.reward(), item.terminated(), nextValues,
                        i * actions, item.nextValidActions(), config.discount());
            }
        }

        float result;
        try (NDManager manager = trainer.getManager().newSubManager()) {
            NDArray input = manager.create(states, new Shape(size, ObservationEncoder.SIZE));
            NDArray action = manager.create(oneHotActions, new Shape(size, actions));
            NDArray expected = manager.create(targets, new Shape(size));
            try (GradientCollector gradients = trainer.newGradientCollector()) {
                NDArray predicted = trainer.forward(new NDList(input)).singletonOrThrow();
                NDArray chosen = predicted.mul(action).sum(new int[]{1});
                NDArray error = chosen.sub(expected).abs();
                NDArray quadratic = error.minimum(1f);
                NDArray loss = quadratic.square().mul(0.5f).add(error.sub(quadratic)).mean();
                result = loss.getFloat();
                if (!Float.isFinite(result)) throw new IllegalStateException("DQN loss became non-finite");
                gradients.backward(loss);
            }
            trainer.step();
        }
        optimizationSteps++;
        if (optimizationSteps % config.targetSyncSteps() == 0) synchronizeTarget();
        return result;
    }

    static float bellmanTarget(float reward, boolean terminated, float[] nextValues,
                               int offset, boolean[] nextValid, float discount) {
        if (terminated) return reward;
        float best = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < nextValid.length; i++) {
            if (nextValid[i]) best = Math.max(best, nextValues[offset + i]);
        }
        if (!Float.isFinite(best)) throw new IllegalArgumentException("Non-terminal transition needs a valid next action");
        return reward + discount * best;
    }

    public void synchronizeTarget() {
        ParameterList source = online.getBlock().getParameters();
        ParameterList destination = target.getBlock().getParameters();
        if (source.size() != destination.size()) throw new IllegalStateException("Network architectures differ");
        for (int i = 0; i < source.size(); i++) {
            String name = source.keyAt(i);
            if (!destination.contains(name)) throw new IllegalStateException("Missing target parameter " + name);
            source.valueAt(i).getArray().copyTo(destination.get(name).getArray());
        }
    }

    private static void checkState(float[] state) {
        if (state == null || state.length != ObservationEncoder.SIZE) {
            throw new IllegalArgumentException("Observation has the wrong dimension");
        }
        for (float value : state) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("Observation contains a non-finite value");
        }
    }

    public int optimizationSteps() { return optimizationSteps; }

    /** Saves inference weights and provenance. Replay and optimizer state are deliberately excluded. */
    public void save(Path file, long trainingSeed, int completedEpisodes) throws IOException {
        if (completedEpisodes < 0) throw new IllegalArgumentException("Negative episode count");
        Path absolute = file.toAbsolutePath();
        Path parent = absolute.getParent();
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, ".neurorescue-", ".tmp");
        try {
            try (DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(temporary)))) {
                output.writeInt(FILE_MAGIC);
                output.writeInt(FILE_VERSION);
                output.writeInt(ObservationEncoder.VERSION);
                output.writeUTF("DQN/PyTorch");
                output.writeUTF(ACTION_ORDER);
                output.writeInt(ObservationEncoder.SIZE);
                output.writeInt(config.hiddenUnits());
                output.writeFloat(config.learningRate());
                output.writeFloat(config.discount());
                output.writeLong(trainingSeed);
                output.writeInt(completedEpisodes);
                output.writeLong(Instant.now().toEpochMilli());
                ParameterList parameters = online.getBlock().getParameters();
                output.writeInt(parameters.size());
                for (int i = 0; i < parameters.size(); i++) {
                    NDArray array = parameters.valueAt(i).getArray();
                    output.writeUTF(parameters.keyAt(i));
                    float[] values = array.toFloatArray();
                    output.writeInt(values.length);
                    for (float value : values) output.writeFloat(value);
                }
            }
            try {
                Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (java.nio.file.AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public static LoadedModel load(Path file) throws IOException {
        try (DataInputStream input = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            if (input.readInt() != FILE_MAGIC || input.readInt() != FILE_VERSION
                    || input.readInt() != ObservationEncoder.VERSION
                    || !"DQN/PyTorch".equals(input.readUTF())
                    || !ACTION_ORDER.equals(input.readUTF())
                    || input.readInt() != ObservationEncoder.SIZE) {
                throw new IOException("Incompatible NeuroRescue model format, encoder, or action space");
            }
            int hidden = input.readInt();
            float learningRate = input.readFloat();
            float discount = input.readFloat();
            long seed = input.readLong();
            int episodes = input.readInt();
            Instant created = Instant.ofEpochMilli(input.readLong());
            if (hidden < 2 || hidden > 512 || episodes < 0) throw new IOException("Invalid model metadata");
            DqnConfig defaults = DqnConfig.defaults();
            DqnConfig config = new DqnConfig(learningRate, discount, defaults.initialEpsilon(),
                    defaults.minimumEpsilon(), defaults.epsilonDecay(), defaults.replayCapacity(),
                    defaults.batchSize(), defaults.warmupSteps(), defaults.targetSyncSteps(),
                    defaults.maxEpisodeSteps(), hidden);
            DqnNetwork network = new DqnNetwork(config, seed);
            try {
                ParameterList parameters = network.online.getBlock().getParameters();
                if (input.readInt() != parameters.size()) throw new IOException("Model parameter count differs");
                try (NDManager manager = network.online.getNDManager().newSubManager()) {
                    for (int i = 0; i < parameters.size(); i++) {
                        NDArray destination = parameters.valueAt(i).getArray();
                        if (!parameters.keyAt(i).equals(input.readUTF())) {
                            throw new IOException("Model parameter names differ");
                        }
                        int length = (int) destination.size();
                        if (input.readInt() != length) throw new IOException("Model parameter shape differs");
                        float[] values = new float[length];
                        for (int j = 0; j < length; j++) {
                            values[j] = input.readFloat();
                            if (!Float.isFinite(values[j])) throw new IOException("Model has non-finite parameters");
                        }
                        manager.create(values, destination.getShape()).copyTo(destination);
                    }
                }
                network.synchronizeTarget();
                return new LoadedModel(network, seed, episodes, created);
            } catch (Exception error) {
                network.close();
                throw new IOException("Could not load NeuroRescue model parameters", error);
            }
        } catch (IllegalArgumentException invalid) {
            throw new IOException("Invalid model metadata", invalid);
        }
    }

    public record LoadedModel(DqnNetwork network, long seed, int completedEpisodes, Instant created) { }

    @Override
    public void close() {
        targetTrainer.close();
        trainer.close();
        target.close();
        online.close();
    }
}
