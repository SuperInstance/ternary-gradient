#![forbid(unsafe_code)]

//! Gradient-free and gradient-like optimization for ternary landscapes.
//!
//! Provides coordinate descent, genetic algorithm, simulated annealing,
//! hill climbing, and fitness landscape traversal on {-1, 0, +1} spaces.

use std::fmt;

/// A ternary value: -1, 0, or +1.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Ternary {
    Neg = -1,
    Zero = 0,
    Pos = 1,
}

impl Ternary {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Ternary::Neg),
            0 => Some(Ternary::Zero),
            1 => Some(Ternary::Pos),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }

    /// All three ternary values.
    pub fn all() -> [Ternary; 3] {
        [Ternary::Neg, Ternary::Zero, Ternary::Pos]
    }

    /// Flip the sign (Neg<->Pos, Zero stays).
    pub fn flip(self) -> Ternary {
        match self {
            Ternary::Neg => Ternary::Pos,
            Ternary::Zero => Ternary::Zero,
            Ternary::Pos => Ternary::Neg,
        }
    }
}

impl fmt::Display for Ternary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i8())
    }
}

/// A point in ternary space (vector of Ternary values).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TernaryPoint {
    coords: Vec<Ternary>,
}

impl TernaryPoint {
    pub fn new(coords: Vec<Ternary>) -> Self {
        Self { coords }
    }

    pub fn from_i8_slice(slice: &[i8]) -> Option<Self> {
        slice.iter().map(|&v| Ternary::from_i8(v)).collect::<Option<Vec<_>>>().map(Self::new)
    }

    pub fn dim(&self) -> usize {
        self.coords.len()
    }

    pub fn coords(&self) -> &[Ternary] {
        &self.coords
    }

    pub fn get(&self, i: usize) -> Option<Ternary> {
        self.coords.get(i).copied()
    }

    pub fn set(&mut self, i: usize, v: Ternary) {
        if i < self.coords.len() {
            self.coords[i] = v;
        }
    }

    /// All neighboring points that differ in exactly one coordinate.
    pub fn neighbors(&self) -> Vec<TernaryPoint> {
        let mut result = Vec::new();
        for i in 0..self.coords.len() {
            for &v in &Ternary::all() {
                if v != self.coords[i] {
                    let mut n = self.clone();
                    n.coords[i] = v;
                    result.push(n);
                }
            }
        }
        result
    }

    /// Hamming distance to another point.
    pub fn hamming_distance(&self, other: &TernaryPoint) -> usize {
        self.coords.iter().zip(other.coords.iter()).filter(|(a, b)| a != b).count()
    }

    /// Random point (seeded by a simple hash of seed).
    pub fn random(dim: usize, seed: u64) -> Self {
        let mut s = seed;
        let mut coords = Vec::with_capacity(dim);
        for _ in 0..dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((s >> 33) % 3) as usize;
            coords.push(Ternary::all()[idx]);
        }
        Self::new(coords)
    }
}

/// A fitness function maps TernaryPoint to f64.
pub type FitnessFn = fn(&TernaryPoint) -> f64;

/// Coordinate descent on ternary space.
pub struct CoordinateDescent {
    dim: usize,
    max_iters: usize,
}

impl CoordinateDescent {
    pub fn new(dim: usize, max_iters: usize) -> Self {
        Self { dim, max_iters }
    }

    pub fn optimize(&self, fitness: FitnessFn, start: &TernaryPoint) -> (TernaryPoint, f64) {
        let mut current = start.clone();
        let mut best_fit = fitness(&current);
        for _ in 0..self.max_iters {
            let mut improved = false;
            for i in 0..current.dim() {
                let original = current.get(i).unwrap();
                let mut best_val = original;
                let mut best_f = best_fit;
                for &v in &Ternary::all() {
                    if v == original { continue; }
                    current.set(i, v);
                    let f = fitness(&current);
                    if f > best_f {
                        best_f = f;
                        best_val = v;
                    }
                }
                current.set(i, best_val);
                if best_f > best_fit {
                    best_fit = best_f;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        (current, best_fit)
    }
}

/// Genetic algorithm with ternary crossover.
pub struct GeneticOptimizer {
    dim: usize,
    pop_size: usize,
    generations: usize,
    seed: u64,
}

impl GeneticOptimizer {
    pub fn new(dim: usize, pop_size: usize, generations: usize, seed: u64) -> Self {
        Self { dim, pop_size, generations, seed }
    }

    fn next_seed(&self, s: &mut u64) -> u64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *s
    }

    pub fn optimize(&self, fitness: FitnessFn) -> (TernaryPoint, f64) {
        let mut rng = self.seed;
        let mut pop: Vec<TernaryPoint> = (0..self.pop_size)
            .map(|_| { let s = self.next_seed(&mut rng); TernaryPoint::random(self.dim, s) })
            .collect();
        let mut best_point = pop[0].clone();
        let mut best_fit = f64::NEG_INFINITY;
        for gen in 0..self.generations {
            // Evaluate
            let scored: Vec<(f64, &TernaryPoint)> = pop.iter().map(|p| (fitness(p), p)).collect();
            for &(f, p) in &scored {
                if f > best_fit {
                    best_fit = f;
                    best_point = p.clone();
                }
            }
            // Selection + crossover
            let mut new_pop = Vec::with_capacity(self.pop_size);
            for _ in 0..self.pop_size {
                let i1 = (self.next_seed(&mut rng) as usize) % self.pop_size;
                let i2 = (self.next_seed(&mut rng) as usize) % self.pop_size;
                let p1 = if scored[i1].0 >= scored[i2].0 { scored[i1].1 } else { scored[i2].1 };
                let i3 = (self.next_seed(&mut rng) as usize) % self.pop_size;
                let i4 = (self.next_seed(&mut rng) as usize) % self.pop_size;
                let p2 = if scored[i3].0 >= scored[i4].0 { scored[i3].1 } else { scored[i4].1 };
                // Crossover
                let mut child = TernaryPoint::random(0, self.next_seed(&mut rng));
                for j in 0..self.dim {
                    let pick = self.next_seed(&mut rng);
                    let c = if pick % 2 == 0 { p1.get(j).unwrap() } else { p2.get(j).unwrap() };
                    child.coords.push(c);
                }
                // Mutation (small chance)
                let muts = self.next_seed(&mut rng) % 100;
                if muts < 10 {
                    let idx = (self.next_seed(&mut rng) as usize) % self.dim;
                    let vals = Ternary::all();
                    child.set(idx, vals[(self.next_seed(&mut rng) as usize) % 3]);
                }
                new_pop.push(child);
            }
            // Elitism
            new_pop[0] = best_point.clone();
            pop = new_pop;
            let _ = gen; // suppress unused
        }
        (best_point, best_fit)
    }
}

/// Simulated annealing with ternary moves.
pub struct SimulatedAnnealing {
    dim: usize,
    initial_temp: f64,
    cooling_rate: f64,
    max_iters: usize,
    seed: u64,
}

impl SimulatedAnnealing {
    pub fn new(dim: usize, initial_temp: f64, cooling_rate: f64, max_iters: usize, seed: u64) -> Self {
        Self { dim, initial_temp, cooling_rate, max_iters, seed }
    }

    pub fn optimize(&self, fitness: FitnessFn, start: &TernaryPoint) -> (TernaryPoint, f64) {
        let mut current = start.clone();
        let mut current_fit = fitness(&current);
        let mut best = current.clone();
        let mut best_fit = current_fit;
        let mut temp = self.initial_temp;
        let mut rng = self.seed;
        for _ in 0..self.max_iters {
            let neighbors = current.neighbors();
            let idx = {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (rng as usize) % neighbors.len()
            };
            let candidate = &neighbors[idx];
            let cand_fit = fitness(candidate);
            let delta = cand_fit - current_fit;
            if delta > 0.0 || (temp > 0.0 && {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let r = (rng as f64) / (u64::MAX as f64);
                r < (delta / temp).exp()
            }) {
                current = candidate.clone();
                current_fit = cand_fit;
                if cand_fit > best_fit {
                    best = current.clone();
                    best_fit = cand_fit;
                }
            }
            temp *= self.cooling_rate;
        }
        (best, best_fit)
    }
}

/// Hill climbing (steepest ascent).
pub struct HillClimbing {
    max_iters: usize,
}

impl HillClimbing {
    pub fn new(max_iters: usize) -> Self {
        Self { max_iters }
    }

    pub fn optimize(&self, fitness: FitnessFn, start: &TernaryPoint) -> (TernaryPoint, f64) {
        let mut current = start.clone();
        let mut current_fit = fitness(&current);
        for _ in 0..self.max_iters {
            let neighbors = current.neighbors();
            let mut best_neighbor = None;
            let mut best_nfit = current_fit;
            for n in &neighbors {
                let f = fitness(n);
                if f > best_nfit {
                    best_nfit = f;
                    best_neighbor = Some(n.clone());
                }
            }
            match best_neighbor {
                Some(n) if best_nfit > current_fit => {
                    current = n;
                    current_fit = best_nfit;
                }
                _ => break,
            }
        }
        (current, current_fit)
    }
}

/// Fitness landscape: evaluates all points and provides traversal.
pub struct FitnessLandscape {
    dim: usize,
    values: Vec<f64>,
}

impl FitnessLandscape {
    /// Create from exhaustive evaluation. `values` indexed by interpreting coords as base-3.
    pub fn new(dim: usize, values: Vec<f64>) -> Self {
        assert_eq!(values.len(), 3usize.pow(dim as u32));
        Self { dim, values }
    }

    /// Build by exhaustive evaluation of a fitness function.
    pub fn from_fn(dim: usize, fitness: FitnessFn) -> Self {
        let total = 3usize.pow(dim as u32);
        let mut values = Vec::with_capacity(total);
        let mut idx = 0;
        Self::enumerate_recursive(dim, &mut Vec::new(), fitness, &mut values, &mut idx);
        Self::new(dim, values)
    }

    fn enumerate_recursive(
        dim: usize,
        coords: &mut Vec<Ternary>,
        fitness: FitnessFn,
        values: &mut Vec<f64>,
        _idx: &mut usize,
    ) {
        if coords.len() == dim {
            let point = TernaryPoint::new(coords.clone());
            values.push(fitness(&point));
            return;
        }
        for &v in &Ternary::all() {
            coords.push(v);
            Self::enumerate_recursive(dim, coords, fitness, values, _idx);
            coords.pop();
        }
    }

    fn coord_to_index(&self, coords: &[Ternary]) -> usize {
        let mut idx = 0;
        for (i, &c) in coords.iter().enumerate() {
            let d = match c {
                Ternary::Neg => 0,
                Ternary::Zero => 1,
                Ternary::Pos => 2,
            };
            idx += d * 3usize.pow((self.dim - 1 - i) as u32);
        }
        idx
    }

    pub fn evaluate(&self, point: &TernaryPoint) -> f64 {
        self.values[self.coord_to_index(point.coords())]
    }

    pub fn global_optimum(&self) -> (TernaryPoint, f64) {
        let (idx, &val) = self.values.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        let mut coords = Vec::with_capacity(self.dim);
        let mut rem = idx;
        for i in (0..self.dim).rev() {
            let base = 3usize.pow(i as u32);
            let d = rem / base;
            rem %= base;
            coords.push(Ternary::all()[d]);
        }
        coords.reverse();
        (TernaryPoint::new(coords), val)
    }

    /// Traverse from start along steepest ascent until a local optimum.
    pub fn steepest_ascent(&self, start: &TernaryPoint) -> (TernaryPoint, f64) {
        let mut current = start.clone();
        let mut current_fit = self.evaluate(&current);
        for _ in 0..1000 {
            let neighbors = current.neighbors();
            let mut best_neighbor = None;
            let mut best_nfit = current_fit;
            for n in &neighbors {
                let f = self.evaluate(n);
                if f > best_nfit {
                    best_nfit = f;
                    best_neighbor = Some(n.clone());
                }
            }
            match best_neighbor {
                Some(n) if best_nfit > current_fit => {
                    current = n;
                    current_fit = best_nfit;
                }
                _ => break,
            }
        }
        (current, current_fit)
    }

    /// Count local optima in the landscape.
    pub fn count_local_optima(&self) -> usize {
        let mut count = 0;
        Self::count_recursive(self.dim, &mut Vec::new(), self, &mut count);
        count
    }

    fn count_recursive(dim: usize, coords: &mut Vec<Ternary>, landscape: &FitnessLandscape, count: &mut usize) {
        if coords.len() == dim {
            let point = TernaryPoint::new(coords.clone());
            let val = landscape.evaluate(&point);
            for n in point.neighbors() {
                if landscape.evaluate(&n) > val {
                    return;
                }
            }
            *count += 1;
            return;
        }
        for &v in &Ternary::all() {
            coords.push(v);
            Self::count_recursive(dim, coords, landscape, count);
            coords.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_ones(p: &TernaryPoint) -> f64 {
        p.coords().iter().filter(|&&v| v == Ternary::Pos).count() as f64
    }

    fn negate_ones(p: &TernaryPoint) -> f64 {
        -(p.coords().iter().filter(|&&v| v == Ternary::Neg).count() as f64)
    }

    fn sum_fn(p: &TernaryPoint) -> f64 {
        p.coords().iter().map(|&v| v.to_i8() as f64).sum()
    }

    #[test]
    fn test_ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Neg));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Zero));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Pos));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn test_ternary_flip() {
        assert_eq!(Ternary::Neg.flip(), Ternary::Pos);
        assert_eq!(Ternary::Pos.flip(), Ternary::Neg);
        assert_eq!(Ternary::Zero.flip(), Ternary::Zero);
    }

    #[test]
    fn test_ternary_display() {
        assert_eq!(format!("{}", Ternary::Neg), "-1");
        assert_eq!(format!("{}", Ternary::Zero), "0");
        assert_eq!(format!("{}", Ternary::Pos), "1");
    }

    #[test]
    fn test_point_from_i8() {
        let p = TernaryPoint::from_i8_slice(&[-1, 0, 1]).unwrap();
        assert_eq!(p.dim(), 3);
        assert_eq!(p.get(0), Some(Ternary::Neg));
    }

    #[test]
    fn test_point_from_i8_invalid() {
        assert!(TernaryPoint::from_i8_slice(&[2]).is_none());
    }

    #[test]
    fn test_point_neighbors() {
        let p = TernaryPoint::from_i8_slice(&[0]).unwrap();
        let n = p.neighbors();
        assert_eq!(n.len(), 2); // can go to -1 or +1
    }

    #[test]
    fn test_point_hamming() {
        let a = TernaryPoint::from_i8_slice(&[-1, 0, 1]).unwrap();
        let b = TernaryPoint::from_i8_slice(&[1, 0, -1]).unwrap();
        assert_eq!(a.hamming_distance(&b), 2);
    }

    #[test]
    fn test_coordinate_descent_finds_optimum() {
        let cd = CoordinateDescent::new(4, 100);
        let start = TernaryPoint::from_i8_slice(&[0, 0, 0, 0]).unwrap();
        let (best, fit) = cd.optimize(count_ones, &start);
        assert_eq!(fit, 4.0);
        assert!(best.coords().iter().all(|&v| v == Ternary::Pos));
    }

    #[test]
    fn test_coordinate_descent_negate() {
        let cd = CoordinateDescent::new(3, 100);
        let start = TernaryPoint::from_i8_slice(&[0, 0, 0]).unwrap();
        let (_, fit) = cd.optimize(negate_ones, &start);
        assert_eq!(fit, 0.0); // best is no negatives
    }

    #[test]
    fn test_hill_climbing_sum() {
        let hc = HillClimbing::new(100);
        let start = TernaryPoint::from_i8_slice(&[0, 0, 0]).unwrap();
        let (best, fit) = hc.optimize(sum_fn, &start);
        assert_eq!(fit, 3.0);
        assert!(best.coords().iter().all(|&v| v == Ternary::Pos));
    }

    #[test]
    fn test_hill_climbing_already_optimal() {
        let hc = HillClimbing::new(100);
        let start = TernaryPoint::from_i8_slice(&[1, 1, 1]).unwrap();
        let (best, fit) = hc.optimize(sum_fn, &start);
        assert_eq!(fit, 3.0);
        assert_eq!(best, start);
    }

    #[test]
    fn test_simulated_annealing_basic() {
        let sa = SimulatedAnnealing::new(4, 10.0, 0.995, 5000, 42);
        let start = TernaryPoint::from_i8_slice(&[0, 0, 0, 0]).unwrap();
        let (_, fit) = sa.optimize(count_ones, &start);
        assert!(fit >= 1.0, "SA should make progress, got {}", fit);
    }

    #[test]
    fn test_genetic_optimizer_basic() {
        let ga = GeneticOptimizer::new(4, 20, 50, 123);
        let (_, fit) = ga.optimize(count_ones);
        assert_eq!(fit, 4.0);
    }

    #[test]
    fn test_genetic_optimizer_sum() {
        let ga = GeneticOptimizer::new(5, 30, 100, 456);
        let (_, fit) = ga.optimize(sum_fn);
        assert_eq!(fit, 5.0);
    }

    #[test]
    fn test_fitness_landscape_evaluate() {
        let landscape = FitnessLandscape::from_fn(2, sum_fn);
        let p = TernaryPoint::from_i8_slice(&[1, 1]).unwrap();
        assert_eq!(landscape.evaluate(&p), 2.0);
    }

    #[test]
    fn test_fitness_landscape_global_optimum() {
        let landscape = FitnessLandscape::from_fn(3, count_ones);
        let (pt, val) = landscape.global_optimum();
        assert_eq!(val, 3.0);
        assert!(pt.coords().iter().all(|&v| v == Ternary::Pos));
    }

    #[test]
    fn test_fitness_landscape_steepest_ascent() {
        let landscape = FitnessLandscape::from_fn(3, sum_fn);
        let start = TernaryPoint::from_i8_slice(&[0, 0, 0]).unwrap();
        let (_, val) = landscape.steepest_ascent(&start);
        assert_eq!(val, 3.0);
    }

    #[test]
    fn test_fitness_landscape_count_local_optima() {
        // sum_fn has exactly 1 local (and global) optimum: all +1
        let landscape = FitnessLandscape::from_fn(2, sum_fn);
        assert!(landscape.count_local_optima() >= 1);
    }

    #[test]
    fn test_random_point_dimensions() {
        let p = TernaryPoint::random(10, 99);
        assert_eq!(p.dim(), 10);
        for &v in p.coords() {
            assert!(v == Ternary::Neg || v == Ternary::Zero || v == Ternary::Pos);
        }
    }

    #[test]
    fn test_point_set() {
        let mut p = TernaryPoint::from_i8_slice(&[0, 0]).unwrap();
        p.set(0, Ternary::Pos);
        assert_eq!(p.get(0), Some(Ternary::Pos));
        p.set(5, Ternary::Neg); // out of bounds, no panic
        assert_eq!(p.dim(), 2);
    }

    #[test]
    fn test_coordinate_descent_max_iters_zero() {
        let cd = CoordinateDescent::new(2, 0);
        let start = TernaryPoint::from_i8_slice(&[-1, -1]).unwrap();
        let (_, fit) = cd.optimize(count_ones, &start);
        assert_eq!(fit, 0.0); // no improvement possible in zero iters
    }

    #[test]
    fn test_ternary_all_values() {
        let all = Ternary::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&Ternary::Neg));
        assert!(all.contains(&Ternary::Zero));
        assert!(all.contains(&Ternary::Pos));
    }
}
