# Future Integration: ternary-gradient

## Current State
Implements gradient-free optimization on {-1, 0, +1} spaces: `TernaryPoint` with Hamming distance and neighbor enumeration, coordinate descent, simulated annealing, hill climbing, and genetic-algorithm-style fitness landscape traversal.

## Integration Opportunities

### With ternary-cell
Every cell's state is a `TernaryPoint`. During the `tick()` cycle's `surprise → vibe → gc` phases, the cell performs implicit gradient descent on its local fitness landscape. `TernaryPoint::neighbors()` enumerates the 2n possible single-trit flips — exactly the mutations a cell should evaluate when deciding whether to change state. The `hill_climb()` and `simulated_annealing()` methods become cell evolution strategies.

### With ternary-rl
Replace the `QTable`'s greedy policy with landscape-aware exploration. `simulated_annealing()` with temperature provides epsilon-greedy exploration with principled cooling. The `FitnessFn` type maps directly to the reward signal from `TernaryEnvironment::step()`.

### With ternary-ga
The `TernaryPoint` type is the genotype. `neighbors()` provides single-gene mutations. Coordinate descent provides local refinement that complements the GA's global search — use GA for population-level exploration, then `coordinate_descent()` for individual refinement.

## Potential in Mature Systems
In room-as-codespace, each room has a fitness landscape over possible ternary configurations. When an ensign agent needs to optimize a room's state (e.g., balance temperature, occupancy, energy), it runs `coordinate_descent()` over the room's ternary state vector. The `FitnessFn` incorporates conservation laws from `construct-core` — configurations that violate conservation constraints get negative fitness. On ESP32, the search collapses to a single-pass hill climb since `neighbors()` is O(n).

## Cross-Pollination Ideas
**Game theory × Gradient:** The fitness landscape of a cooperative game IS the gradient landscape. Nash equilibria are saddle points. `simulated_annealing()` with game-theoretic fitness finds mixed-strategy equilibria in ternary games.

**Topology × Gradient:** Morse theory connects gradient flows to topology. The critical points of ternary fitness landscapes (where all neighbors have equal/lower fitness) correspond to topological features of the ternary state space. `ternary-topology` could classify these.

## Dependencies for Next Steps
- Integration with `ternary-cell` requires a `FitnessFn` that wraps the cell's `surprise()` computation
- `ternary-tensor` for batched fitness evaluation across large cell populations
- Benchmark: does ternary gradient descent converge on the same optima as floating-point SGD?
