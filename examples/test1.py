import fiberv2 as fiber
from fiberv2._groups.so3 import Rotation3d, Spin3d

a = Rotation3d.from_euler("xyz", [0, 90, 0])
b = Spin3d.from_vector([1, 0, 0])

print(b * b)
