# 0001: Keep the first vertical slice small and headless-testable

**Status:** Accepted, 2026-10-27

The Gradle build choice in this historical record was superseded by [decision 0002](0002-maven-migration.md). The simulation architecture remains unchanged.

The source repository contained a Maven JavaFX hello-world application and no simulation code. We replaced the starter build with Gradle 9.3.0, which supports the installed JDK 25, and retained JavaFX 21.0.6 from the starter dependency line. Java 21 language compatibility keeps the source accessible to LTS users.

The first slice uses one module. The simulation package has no JavaFX dependency, so domain tests run without a display. Built-in maps are static and validated; hazardous tiles are terminal. A shortest safe route baseline checks that the mission rules can be completed. A trainable neural agent is deliberately deferred until its learning behavior can be measured. This avoids giving users a UI that implies training before a real training loop exists.
