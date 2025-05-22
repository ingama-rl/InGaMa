from typing import Any, Optional, Tuple, Union, List

import chex
from distrax._src.distributions import distribution
from distrax._src.distributions.categorical import _kl_divergence_categorical_categorical
from distrax._src.utils import math
from distrax import Categorical
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

Array = chex.Array
PRNGKey = chex.PRNGKey
EventT = distribution.EventT

class MultiCategorical(distribution.Distribution):
    """Categorical distribution."""

    def __init__(self,
                 categoricals: List[Categorical]):
        """Initializes a Categorical distribution.

        Args:
          logits: Logit transform of the probability of each category. Only one
            of `logits` or `probs` can be specified.
          probs: Probability of each category. Only one of `logits` or `probs` can
            be specified.
          dtype: The type of event samples.
        """
        super().__init__()
        self._cats = [Categorical(cat) for cat in categoricals]

    @property
    def event_shape(self) -> Tuple[int, ...]:
        """Shape of event of distribution samples."""
        return (len(self._cats),)

    @property
    def logits(self) -> Array:
        """The logits for each event."""
        return jnp.array([cat.logits for cat in self._cats])

    @property
    def probs(self) -> Array:
        """The probabilities for each event."""
        return jnp.array([cat.probs for cat in self._cats])

    @property
    def num_categoricals(self) -> int:
        """Number of categories."""
        return len(self._cats)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        """See `Distribution._sample_n`."""
        smpls = []
        for cat in self._cats:
            key, subkey = jax.random.split(key)
            smpls.append(cat._sample_n(subkey, n)[..., None])#.reshape(n, -1, 1))
        return jnp.concatenate(smpls, axis=-1)#.reshape((n, -1, len(self._cats)))

    def log_prob(self, value: EventT) -> Array:
        """See `Distribution.log_prob`."""
        log_prob = 0
        chunks = jnp.split(value, self.num_categoricals, axis=-1)
        for value, cat in zip(chunks, self._cats):
            log_prob = log_prob + cat.log_prob(value.squeeze(axis=-1))
        return log_prob

    def prob(self, value: EventT) -> Array:
        """See `Distribution.prob`."""
        prob = 1
        for i, cat in enumerate(self._cats):
            prob = prob * cat.prob(value[i])
        return prob

    def entropy(self) -> Array:
        """See `Distribution.entropy`."""
        entropy = 0
        for i, cat in enumerate(self._cats):
            entropy = entropy + cat.entropy()
        return entropy

    def mode(self) -> Array:
        """See `Distribution.mode`."""
        return jnp.concatenate([cat.mode() for cat in self._cats], axis=-1)

    def logits_parameter(self) -> Array:
        """Wrapper for `logits` property, for TFP API compatibility."""
        return jnp.concatenate([cat.logits for cat in self._cats], axis=-1)

    def __getitem__(self, index) -> 'Categorical':
        """See `Distribution.__getitem__`."""
        index = distribution.to_batch_shape_index(self.batch_shape, index)
        return MultiCategorical([cat.__getitem__(index[i]) for i, cat in enumerate(self._cats)])


def _kl_divergence_multicategorical(
        dist1: MultiCategorical,
        dist2: MultiCategorical,
        *unused_args, **unused_kwargs,
) -> Array:
    """Obtains the KL divergence `KL(dist1 || dist2)` between two Categoricals.

    The KL computation takes into account that `0 * log(0) = 0`; therefore,
    `dist1` may have zeros in its probability vector.

    Args:
      dist1: A Categorical distribution.
      dist2: A Categorical distribution.

    Returns:
      Batchwise `KL(dist1 || dist2)`.

    Raises:
      ValueError if the two distributions have different number of categories.
    """
    num_categories1 = dist1.num_categoricals
    num_categories2 = dist2.num_categoricals

    if num_categories1 != num_categories2:
        raise ValueError(
            f'Cannot obtain the KL between two Categorical distributions '
            f'with different number of categories: the first distribution has '
            f'{num_categories1} categories, while the second distribution has '
            f'{num_categories2} categories.')
    return jnp.concatenate(
        [_kl_divergence_categorical_categorical(cat1, cat2) for cat1, cat2 in zip(dist1._cats, dist2._cats)],
        axis=-1
    ).sum(axis=-1)


