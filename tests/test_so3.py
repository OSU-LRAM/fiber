import fiber
from fiber import Rotation3d


def test_adj():
    g = Rotation3d.from_euler("xyz", [90, 0, 0])
    h = Rotation3d.from_euler("xyz", [0, 90, 0])
    dt = 0.05  # s

    g @ fiber.logm(h) * dt


def test_expm(): ...
