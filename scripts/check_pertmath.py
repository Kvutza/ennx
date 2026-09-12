"""Deterministic ideal Gaussian moment checks; no GPU, BF16, or model claims."""

from itertools import product
from math import fsum, isclose, prod, sqrt

NODES = ((-sqrt(3), 1 / 6), (0.0, 2 / 3), (sqrt(3), 1 / 6))
RHOS = (0.0, 0.25, 0.75, 0.99, 1.0)
# Exactly BF16-representable values, including zeros and uneven magnitudes.
REFERENCES = ((-2.0,), (0.0, 0.125), (16.0, -0.25, 0.0), (0.5, -2.0, 4.0))


def close(actual, expected):
    if not isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12):
        raise AssertionError(f"{actual!r} != {expected!r}")


def quadrature(dimension):
    """Product rule exact through degree five per coordinate in real arithmetic."""
    return tuple(
        (tuple(node for node, _ in sample), prod(mass for _, mass in sample))
        for sample in product(NODES, repeat=dimension)
    )


def expectation(distribution, function):
    return fsum(mass * function(value) for value, mass in distribution)


def moments(distribution, function):
    values = tuple((function(value), mass) for value, mass in distribution)
    mean = expectation(values, lambda value: value)
    variance = expectation(values, lambda value: (value - mean) ** 2)
    return mean, variance


def normalize(reference):
    q = fsum(value**2 for value in reference) / len(reference)
    if q <= 0:
        raise ValueError("reference tensor must be nonzero")
    return tuple(value / sqrt(q) for value in reference)


def direction(reference, rho, noise):
    return tuple(
        rho * value + sqrt(1 - rho**2) * fresh
        for value, fresh in zip(reference, noise, strict=True)
    )


def check_quadrature():
    distribution = quadrature(1)
    for degree, expected in enumerate((1, 0, 1, 0, 3, 0)):
        close(
            expectation(distribution, lambda e, degree=degree: e[0] ** degree), expected
        )
    pair = quadrature(2)
    close(expectation(pair, lambda e: e[0] * e[1]), 0)
    close(expectation(pair, lambda e: e[0] ** 2 * e[1] ** 2), 1)


def check_radius():
    for stored in REFERENCES:
        reference = normalize(stored)
        n = len(reference)
        distribution = quadrature(n)
        close(fsum(value**2 for value in reference) / n, 1)
        for rho in RHOS:
            proposals = tuple(
                (direction(reference, rho, noise), mass) for noise, mass in distribution
            )
            energies = tuple(
                (fsum(u**2 for u in values) / n, mass) for values, mass in proposals
            )
            mean, variance = moments(
                energies,
                lambda energy: energy,
            )
            close(mean, 1)
            close(variance, 2 * (1 - rho**4) / n)
            close(
                fsum(
                    mass
                    * fsum(u * v for u, v in zip(values, reference, strict=True))
                    / n
                    for values, mass in proposals
                ),
                rho,
            )
            for radius, scale in ((0.005, 0.125), (0.02, 4.0), (0.5, 2.0)):
                delta_mean, delta_variance = moments(
                    tuple(
                        (fsum((radius * scale * u) ** 2 for u in values) / n, mass)
                        for values, mass in proposals
                    ),
                    lambda energy: energy,
                )
                close(delta_mean, (radius * scale) ** 2)
                close(delta_variance, (radius * scale) ** 4 * variance)


def check_distance():
    # Two tensor blocks: normalization is per tensor, not across all coordinates.
    blocks = ((0.0, 0.125), (-2.0,))
    reference = tuple(v for block in blocks for v in normalize(block))
    base = (2.0, -1.0, 4.0)
    scales = (0.5, 0.5, 3.0)
    weights = (2.0, 2.0, 1 / 3)
    distribution = quadrature(len(reference))
    for history in (base, (0.0, 3.0, 1.0), (-4.0, 0.5, 16.0)):
        for radius, rho in product((0.005, 0.02, 0.4), RHOS):
            m = tuple(
                b - h + radius * s * rho * v
                for b, h, s, v in zip(base, history, scales, reference, strict=True)
            )
            sigma = tuple(radius * s * sqrt(1 - rho**2) for s in scales)

            def distance(noise, rho=rho, radius=radius, history=history):
                u = direction(reference, rho, noise)
                return fsum(
                    w * (b + radius * s * value - h) ** 2
                    for w, b, s, value, h in zip(
                        weights, base, scales, u, history, strict=True
                    )
                )

            mean, variance = moments(distribution, distance)
            close(
                mean,
                fsum(
                    w * (a**2 + b**2) for w, a, b in zip(weights, m, sigma, strict=True)
                ),
            )
            close(
                variance,
                fsum(
                    w**2 * (2 * b**4 + 4 * b**2 * a**2)
                    for w, a, b in zip(weights, m, sigma, strict=True)
                ),
            )
            if rho < 1 and variance <= 0:
                raise AssertionError("expected distance must not erase seed variation")


def check_directions():
    reference = normalize((0.5, -2.0))
    n = len(reference)
    distribution = quadrature(n)
    small, large = 0.005, 0.02
    for rho in (0.0, 0.75):
        # Shared noise: U is identical within a radius pair, not independent.
        actual = expectation(
            distribution,
            lambda e, rho=rho: fsum(
                (small * u - large * u) ** 2 for u in direction(reference, rho, e)
            ),
        )
        close(actual, n * (small - large) ** 2)

    # Across pairs the two Gaussian fields are independent, conditional on R.
    actual = expectation(
        quadrature(2 * n),
        lambda e: fsum(
            (small * a - large * b) ** 2
            for a, b in zip(
                direction(reference, 0.75, e[:n]),
                direction(reference, 0.0, e[n:]),
                strict=True,
            )
        ),
    )
    close(actual, n * (small**2 + large**2))


if __name__ == "__main__":
    check_quadrature()
    check_radius()
    check_distance()
    check_directions()
    print(
        "PASS: Gaussian quadrature, conditional radius/distance moments, paired noise"
    )
