import abc
import numpy as np

class Sampler(abc.ABC):
    """Abstract Base Class for Samplers."""

    def __init__(self, random_state: int | np.random.Generator | None = None):
        """
        Initializes the sampler with an optional random_state.

        Args:
            random_state (int | np.random.Generator | None): Random seed or Generator.
        """
        self.random_state = random_state
        if isinstance(random_state, np.random.Generator):
            self.rng = random_state
        else:
            self.rng = np.random.default_rng(random_state)

    @abc.abstractmethod
    def sample(self, n: int, dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Generates n samples in a dim-dimensional unit hypercube [0, 1]^dim.

        Args:
            n (int): Number of samples to generate.
            dim (int): Dimension of the space.
            rng (np.random.Generator | None): Optional generator. If None, uses internal self.rng.

        Returns:
            np.ndarray: Array of shape (n, dim) with samples in [0, 1]^dim.
        """
        pass


class LHCSampler(Sampler):
    """Latin Hypercube Sampler (LHS) in pure NumPy."""

    def sample(self, n: int, dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if n <= 0:
            return np.empty((0, dim), dtype=np.float32)

        active_rng = rng if rng is not None else self.rng
        perms = np.empty((n, dim), dtype=np.float32)
        for d in range(dim):
            perms[:, d] = active_rng.permutation(n)

        offsets = active_rng.uniform(0.0, 1.0, (n, dim)).astype(np.float32)
        return (perms + offsets) / n


class UniformSampler(Sampler):
    """Uniform Random Sampler in pure NumPy."""

    def sample(self, n: int, dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if n <= 0:
            return np.empty((0, dim), dtype=np.float32)

        active_rng = rng if rng is not None else self.rng
        return active_rng.uniform(0.0, 1.0, (n, dim)).astype(np.float32)


def check_sampler(
    sampler: Sampler | int | None,
    default_random_state: int | np.random.Generator | None = None
) -> Sampler:
    """
    Validates and returns a Sampler instance.

    If sampler is None, returns LHCSampler(random_state=default_random_state).
    If sampler is an int/numeric, returns LHCSampler(random_state=sampler).
    If sampler is already a Sampler instance, returns it.

    Args:
        sampler (Sampler | int | None): Sampler configuration.
        default_random_state (int | np.random.Generator | None): Default seed to use if sampler is None.

    Returns:
        Sampler: Checked Sampler instance.
    """
    if sampler is None:
        return LHCSampler(random_state=default_random_state)
    if isinstance(sampler, (int, np.integer)):
        return LHCSampler(random_state=int(sampler))
    if isinstance(sampler, Sampler):
        return sampler
    raise TypeError(f"Invalid sampler type: {type(sampler)}. Must be None, int, or an instance of Sampler.")
