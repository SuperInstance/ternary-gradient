# ternary-gradient

Gradient-free and gradient-like optimization on ternary landscapes — coordinate descent, genetic algorithms, simulated annealing, and hill climbing over {-1, 0, +1} spaces.

## Why This Exists

Most optimization tooling assumes continuous, differentiable landscapes. But when your decision variables are inherently ternary — approve/abstain/reject, positive/neutral/negative, on/off/standby — you need optimizers that work natively in discrete three-valued space. This crate provides a suite of classic optimization algorithms that operate directly on ternary vectors, plus a `FitnessLandscape` type for exhaustive enumeration and analysis. `forbid(unsafe_code)` throughout.

## Core Concepts

- **TernaryPoint**: A point in ternary space — a vector of `Ternary` values with neighbor enumeration and Hamming distance.
- **CoordinateDescent**: Iteratively optimize each coordinate independently; converges to local optima.
- **GeneticOptimizer**: Population-based search with tournament selection, uniform crossover, and mutation on ternary genes.
- **SimulatedAnnealing**: Probabilistic neighbor exploration with temperature decay — escapes local optima.
- **HillClimbing**: Steepest-ascent hill climbing over the neighbor graph.
- **FitnessLandscape**: Exhaustive evaluation of all `3ⁿ` points for small dimensions; global optimum finding, local optima counting, steepest-ascent traversal.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-gradient = "0.1"
```

```rust
use ternary_gradient::{
    Ternary, TernaryPoint, CoordinateDescent, HillClimbing,
    SimulatedAnnealing, GeneticOptimizer, FitnessLandscape,
};

fn count_pos(p: &TernaryPoint) -> f64 {
    p.coords().iter().filter(|&&v| v == Ternary::Pos).count() as f64
}

fn main() {
    let start = TernaryPoint::from_i8_slice(&[0, 0, 0, 0]).unwrap();

    // Coordinate descent
    let cd = CoordinateDescent::new(4, 100);
    let (best, fit) = cd.optimize(count_pos, &start);
    assert_eq!(fit, 4.0);

    // Hill climbing
    let hc = HillClimbing::new(100);
    let (best, fit) = hc.optimize(count_pos, &start);

    // Simulated annealing
    let sa = SimulatedAnnealing::new(4, 10.0, 0.995, 5000, 42);
    let (best, fit) = sa.optimize(count_pos, &start);

    // Genetic algorithm
    let ga = GeneticOptimizer::new(4, 20, 50, 123);
    let (best, fit) = ga.optimize(count_pos);

    // Exhaustive landscape (small dimensions only)
    let landscape = FitnessLandscape::from_fn(3, count_pos);
    let (global_best, val) = landscape.global_optimum();
    println!("Global optimum: {:?}, value: {}", global_best.coords(), val);
}
```

## API Overview

| Type | Description |
|---|---|
| `Ternary` | Value: `Neg`, `Zero`, `Pos` |
| `TernaryPoint` | Vector of ternary values with `neighbors()`, `hamming_distance()`, `random()` |
| `CoordinateDescent` | Iterative coordinate-wise optimization |
| `GeneticOptimizer` | GA with tournament selection, crossover, mutation, elitism |
| `SimulatedAnnealing` | Temperature-based probabilistic neighbor search |
| `HillClimbing` | Steepest ascent on the neighbor graph |
| `FitnessLandscape` | Exhaustive `3ⁿ` evaluation with `global_optimum()`, `steepest_ascent()`, `count_local_optima()` |
| `FitnessFn` | Type alias: `fn(&TernaryPoint) -> f64` |

## How It Works

Each optimizer operates on `TernaryPoint` — a vector where each coordinate is one of three values. The **neighbor graph** connects each point to all points that differ in exactly one coordinate (2 neighbors per dimension), giving `2n` neighbors for an `n`-dimensional point.

**CoordinateDescent** cycles through dimensions, trying all three values and picking the best. **HillClimbing** evaluates all neighbors and moves to the best. **SimulatedAnnealing** randomly samples neighbors and accepts worse moves with probability `e^(Δ/T)`. **GeneticOptimizer** maintains a population, selects parents via tournament, produces children via uniform crossover, and applies random mutations with 10% probability per child.

`FitnessLandscape` enumerates all `3ⁿ` points (practical for n ≤ ~12) and stores fitness values in a flat array indexed by base-3 interpretation of coordinates.

## Use Cases

- **Portfolio optimization**: Each asset gets a ternary weight (underweight / neutral / overweight).
- **Circuit design**: Optimize ternary logic gate configurations where each input is {-1, 0, +1}.
- **Game AI**: Search strategy spaces where actions are inherently three-valued (retreat / hold / advance).
- **Feature selection with ternary encoding**: Each feature is negatively selected, neutral, or positively selected.

## Ecosystem

Part of the **SuperInstance** ternary computing suite:

- `ternary-lattice` — lattice structures for ternary values
- `ternary-codes` — error-correcting codes for ternary data
- `ternary-gradient` — this crate
- `ternary-language` — ternary NLP and grammar processing
- `ternary-trees` — ternary decision trees and forests
- `ternary-transform` — wavelet, Fourier, and kernel transforms
- `ternary-planning` — planning and scheduling with ternary priorities
- `ternary-rl` — reinforcement learning with ternary actions
- `ternary-som` — self-organizing maps for ternary data
- `ternary-failure` — failure analysis with ternary classification

## License

MIT

## See Also
- **ternary-fitness** — related
- **ternary-compass** — related
- **ternary-ga** — related
- **ternary-optimization** — related
- **ternary-energy** — related

