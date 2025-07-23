from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.profiles import PowerSeriesProfile
import desc
import matplotlib.pyplot as plt

from desc.plotting import plot_surfaces

# eq = desc.examples.get("W7-X")
# eq1, info = eq.solve(verbose=3, copy=True)

surf = FourierRZToroidalSurface(
    R_lmn=[10.0, -1.0, -0.3, 0.3],
    modes_R=[
        (0, 0),
        (1, 0),
        (1, 1),
        (-1, -1),
    ],  # (m,n) pairs corresponding to R_mn on previous line
    Z_lmn=[1, -0.3, -0.3],
    modes_Z=[(-1, 0), (-1, 1), (1, -1)],
    NFP=19,
)
pressure = PowerSeriesProfile([1.8e4, 0, -3.6e4, 0, 1.8e4])
iota = PowerSeriesProfile([1, 0, 1.5])  # 1 + 1.5 r^2

eq = Equilibrium(
    L=8,  # radial resolution
    M=8,  # poloidal resolution
    N=3,  # toroidal resolution
    surface=surf,
    pressure=pressure,
    iota=iota,
    Psi=1.0,  # total flux, in Webers
)

eq1, info = eq.solve(verbose=3, copy=True)

plot_surfaces(eq1)
plt.show()



# import traceback
# stack = traceback.extract_stack()
# filename, lineno, func, text = stack[-2]  # Get the second to last entry (the caller)
# print(f"Called from: {filename}:{lineno}")

# from desc.basis