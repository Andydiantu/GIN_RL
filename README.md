# Reinforcement Learning for Context-Aware Mutation Selection in LLM-Based Genetic Improvement

This repository contains the source-code artifact accompanying the paper **"Reinforcement Learning for Context-Aware Mutation Selection in LLM-Based Genetic Improvement."** The implementation extends [Gin](https://github.com/gintool/gin) with RL based online contextual mutation-operator selection, LLM-based mutation operators, and LLM-guided destination selection.


## Getting Started

### Prerequisites

The artifact requires:

1. **JDK 17**.
2. **Ollama** when using a local model.
3. **Maven** when the target programme is a Maven project.

### Build and Test

From the repository root, run:

```bash
./gradlew test shadowJar
```

This runs the focused tests for the paper-specific Bayesian models and creates the executable fat JAR:

```text
build/gin.jar
```

## Setting Up a Local LLM

Start Ollama and download the desired model. For example:

```bash
ollama serve
ollama pull gemma3:4b
```

## Setting Up a Benchmark

The paper evaluates five Java projects:

| Project | Repository | Revision used in the study |
|---|---|---|
| JCodec | [jcodec/jcodec](https://github.com/jcodec/jcodec) | `7e52834` |
| JUnit4 | [junit-team/junit4](https://github.com/junit-team/junit4) | `r4.13.2` |
| Gson | [google/gson](https://github.com/google/gson) | `gson-parent-2.10.1` |
| Commons Net | [apache/commons-net](https://github.com/apache/commons-net) | `commons-net-3.10.0` |
| Karate | [karatelabs/karate](https://github.com/karatelabs/karate) | `v1.4.1` |


### Profiling a Maven Project

The samplers consume a profiler CSV that identifies target methods and their associated tests. A typical profiling command is:

```bash
java -cp build/gin.jar gin.util.Profiler \
  -r 20 \
  -p <project-name> \
  -d <project-directory> \
  -h <maven-home> \
  -o <profiler-output.csv>
```

For example, on a standard Linux installation, `<maven-home>` is commonly `/usr/share/maven`. Profiling support inherits Gin's Java Flight Recorder and build-system constraints.

## Running Experiments

The commands below assume the following shell variables:

```bash
GIN_JAR=/path/to/gin-paper-upload/build/gin.jar
PROJECT=<project-name>
PROJECT_DIR=/path/to/target-project
PROFILE=/path/to/profiler-output.csv
MAVEN_HOME=/usr/share/maven
MODEL=gemma3:4b
```

### RL+DS Random Search

```bash
java -Dtinylog.level=info -cp "$GIN_JAR" gin.util.RandomSampler \
  -j -p "$PROJECT" -d "$PROJECT_DIR" -m "$PROFILE" \
  -o rl_ds_random.csv -to rl_ds_random_timing.csv \
  -h "$MAVEN_HOME" -x 10000 \
  -et gin.edit.llm.LLMReplaceStatement,gin.edit.llm.LLMMaskedStatement,STATEMENT \
  -mt "$MODEL" -pt MASKED -mo 300 -pn 1000 \
  -rl true -ds true -fm FULL_INTERACTION
```

### Masking Random Search

```bash
java -Dtinylog.level=info -cp "$GIN_JAR" gin.util.RandomSampler \
  -j -p "$PROJECT" -d "$PROJECT_DIR" -m "$PROFILE" \
  -o masking_random.csv -to masking_random_timing.csv \
  -h "$MAVEN_HOME" -x 10000 \
  -et gin.edit.llm.LLMMaskedStatement \
  -mt "$MODEL" -pt MASKED -mo 300 -pn 1000
```

### Replacement Random Search

```bash
java -Dtinylog.level=info -cp "$GIN_JAR" gin.util.RandomSampler \
  -j -p "$PROJECT" -d "$PROJECT_DIR" -m "$PROFILE" \
  -o replacement_random.csv -to replacement_random_timing.csv \
  -h "$MAVEN_HOME" -x 10000 \
  -et gin.edit.llm.LLMReplaceStatement \
  -mt "$MODEL" -pt WITH_MUTATION_EXAMPLES -mo 300 -pn 1000
```

### Traditional Random Search

This baseline uniformly samples the traditional statement-level operators and does not query an LLM.

```bash
java -Dtinylog.level=info -cp "$GIN_JAR" gin.util.RandomSampler \
  -j -p "$PROJECT" -d "$PROJECT_DIR" -m "$PROFILE" \
  -o traditional_random.csv -to traditional_random_timing.csv \
  -h "$MAVEN_HOME" -x 10000 \
  -et STATEMENT -pn 1000
```

### RL+DS Interleaved Local Search

Local search uses the same contextual operator-selection and destination-selection configuration. `-in` is the number of evaluations allocated to each target method, including its baseline evaluation. For 10 methods and `-in 100`, the total budget is 1,000 evaluations.

```bash
java -Dtinylog.level=info -cp "$GIN_JAR" gin.util.LocalSearchRuntime \
  -j -p "$PROJECT" -d "$PROJECT_DIR" -m "$PROFILE" \
  -o rl_ds_local.csv -to rl_ds_local_timing.csv \
  -h "$MAVEN_HOME" -x 10000 \
  -et gin.edit.llm.LLMReplaceStatement,gin.edit.llm.LLMMaskedStatement,STATEMENT \
  -mt "$MODEL" -pt MASKED -mo 300 -in 100 \
  -rl true -ds true -fm FULL_INTERACTION
```

The local-search implementation preserves a separate incumbent patch for each target method and shuffles the fixed per-method evaluation allocations before execution. Thus, interleaving changes only the order in which methods are visited, not their individual budgets.

## Important Arguments

| Argument | Meaning |
|---|---|
| `-d` | Target-project directory. |
| `-m` | Profiler CSV containing target methods and tests. |
| `-p` | Target-project name. |
| `-h` | Maven home directory. |
| `-j` | Evaluate patches in a separate JVM. |
| `-x` | Per-test timeout in milliseconds. |
| `-et` | Comma-separated mutation-operator classes or an edit family such as `STATEMENT`. |
| `-mt` | Ollama model tag, `OpenAI`, or `DeepSeek`. |
| `-mo` | LLM request timeout in seconds. |
| `-pt` | LLM prompt type. |
| `-pn` | Number of random-search patches. |
| `-in` | Number of local-search evaluations per target method. |
| `-rl` | Enable online RL mutation-operator selection. |
| `-ds` | Enable LLM destination selection for traditional operators. |
| `-fm` | Bandit feature map: `ADDITIVE`, `FULL_INTERACTION`, or `UNCONTEXTUAL`. |
| `-o` | Patch-level and test-level result CSV. |
| `-to` | End-to-end timing output CSV. |


## Output

The result CSV contains one row per evaluated test, including the patch, selected operator, validity, compilation status, test result, execution time, reward, and band index. Aggregate patch outcomes by `PatchIndex`; a patch passes only if all of its associated test rows pass.

Logs can be captured using ordinary shell redirection when required.

## License

See `LICENSE.md`.
