# 0002: Use Maven as the sole build system

**Status:** Accepted, 2026-10-27

Maven is the project's sole build system. Its standard source layout and explicit plugin configuration keep compilation, testing, packaging, and JavaFX launch in one readable `pom.xml`. Maven Wrapper 3.3.4 pins Maven 3.9.16, so contributors can run the same commands without a global Maven installation. OS-activated dependencies supply the PyTorch CPU binaries for supported Windows and Linux x86-64 systems.

The migration from Gradle preserved the non-modular source layout, application entry point, Java 21 release target, JavaFX 21.0.6, and JUnit Jupiter 5.12.1. It changed build files, documentation, and CI rather than application behavior. The compiler and Surefire plugins build and test the headless code; `clean verify` produces a standard JAR. The JavaFX plugin supplies the runtime dependencies for `javafx:run`. The longer learning experiment remains an opt-in test.
