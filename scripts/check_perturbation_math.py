"""Exact finite checks for docs/perturbations.md; no model evaluations."""

from fractions import Fraction as F
from itertools import product


def children(parent, probability):
    for flips in product((False, True), repeat=len(parent)):
        signs = tuple(-s if flip else s for s, flip in zip(parent, flips))
        mass = F(1)
        for flip in flips:
            mass *= probability if flip else 1 - probability
        yield signs, mass


def expectation(distribution, function):
    return sum((mass * function(value) for value, mass in distribution), F(0))


def squared_distance(x, y, weights):
    return sum((w * (a - b) ** 2 for a, b, w in zip(x, y, weights)), F(0))


def check_tensor_radius():
    base = (F(1), F(7))  # RMS = 5 exactly.
    radius, scale = F(1, 8), F(5)
    expected = radius**2 * sum(w**2 for w in base)
    for signs in product((-1, 1), repeat=len(base)):
        delta = tuple(radius * scale * s for s in signs)
        assert sum(d**2 for d in delta) == expected


def check_distance_moments():
    parent = (1, -1, 1)
    base = (F(2), F(-1), F(4))
    history = (F(0), F(3), F(1))
    scales = (F(1, 2), F(2), F(3))
    weights = (F(2), F(1, 3), F(1))
    radius = F(2, 5)
    for probability in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        rho = 1 - 2 * probability
        distribution = tuple(children(parent, probability))
        assert sum(mass for _, mass in distribution) == 1
        if 0 < probability < 1:
            assert all(mass > 0 for _, mass in distribution)
        for i, sign in enumerate(parent):
            assert expectation(distribution, lambda t: sign * t[i]) == rho

        distances = []
        for signs, mass in distribution:
            candidate = tuple(b + radius * s * t for b, s, t in zip(base, scales, signs))
            distances.append((squared_distance(candidate, history, weights), mass))
        mean = expectation(distances, lambda d: d)
        variance = expectation(distances, lambda d: (d - mean) ** 2)
        expected_mean = sum(
            w * ((b - h) ** 2 + 2 * radius * s * rho * t * (b - h) + radius**2 * s**2)
            for w, b, h, s, t in zip(weights, base, history, scales, parent)
        )
        expected_variance = 4 * radius**2 * (1 - rho**2) * sum(
            w**2 * s**2 * (b - h) ** 2
            for w, s, b, h in zip(weights, scales, base, history)
        )
        assert mean == expected_mean
        assert variance == expected_variance
        if probability == F(1, 2):
            # A common expected distance does not determine realized seed ranking.
            assert len({distance for distance, _ in distances}) > 1
            assert variance > 0


def check_sibling_distance():
    parent = (1, -1, 1)
    p, q = F(1, 4), F(1, 3)
    r, t = F(1, 2), F(3, 4)
    rho, sigma = 1 - 2 * p, 1 - 2 * q
    actual = sum(
        px * py * sum((r * a - t * b) ** 2 for a, b in zip(x, y))
        for x, px in children(parent, p)
        for y, py in children(parent, q)
    )
    assert actual == len(parent) * (r**2 + t**2 - 2 * r * t * rho * sigma)


def check_reward_covariance():
    # Mean 7, with variance 4, 9, and 16 at degrees one, two, and three.
    def reward(s):
        return 7 + 2 * s[0] + 3 * s[0] * s[1] + 4 * s[0] * s[1] * s[2]

    for probability in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
        rho = 1 - 2 * probability
        covariance = sum(
            mass * (reward(s) - 7) * (reward(t) - 7) / 8
            for s in product((-1, 1), repeat=3)
            for t, mass in children(s, probability)
        )
        assert covariance == 4 * rho + 9 * rho**2 + 16 * rho**3


if __name__ == "__main__":
    check_tensor_radius()
    check_distance_moments()
    check_sibling_distance()
    check_reward_covariance()
    print("PASS: tensor radius, distance moments, sibling distance, reward covariance")
