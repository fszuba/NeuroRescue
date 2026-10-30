package it.farrlor.neurorescue;

enum TrainingStatus {
    IDLE, STARTING, TRAINING, PAUSED, COMPLETED, STOPPED, FAILED;

    String label(int completed, int requested) {
        return switch (this) {
            case IDLE -> "Idle";
            case STARTING -> "Starting · " + completed + " / " + requested + " episodes";
            case TRAINING -> "Training on CPU · " + completed + " / " + requested + " episodes";
            case PAUSED -> "Paused · " + completed + " / " + requested + " episodes";
            case COMPLETED -> "Completed · " + completed + " / " + requested + " episodes";
            case STOPPED -> "Stopped · " + completed + " / " + requested + " episodes";
            case FAILED -> "Training failed · See error details";
        };
    }
}
