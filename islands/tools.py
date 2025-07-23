import numpy as np
from desc.coils import FourierXYZCoil, CoilSet
from desc.magnetic_fields import field_line_integrate
from simsopt.configs import get_w7x_data

def get_w7x_coils_desc():
    """Returns the W7-X coils in a DESC CoilSet format.
    
        Returns
        -------
        desc_coils : CoilSet
            W7-X coils in DESC CoilSet format
        """

    # load in coils from simsopt
    simsopt_curves, simsopt_currents, _ = get_w7x_data()

    # convert the base curves from a simsopt representation to a desc 
    # representation. 
    # 
    # in simsopt, the curves are described as an xyz fourier series of the form 
    #                   x(θ)=∑x_cm*cos(mθ)+x_sm*sin(mθ), 
    # where the dofs are stored as
    #           [x_c0, x_s1, x_c1, ..., x_sN, x_cN, y_c0, y_s1, y_c1,...].
    #
    # by contrast, in desc the curves are described as an xyz fourier series of the 
    # form 
    #               x(θ)={x_n*cos(|n|θ) for n≧0, x_n*sin(|n|θ) for n<0}.
    # where the dofs have a more arbitrary storage structure.
    desc_coils = []
    for curve, current in zip(simsopt_curves, simsopt_currents):
        simsopt_dofs = curve.get_dofs()
        nmodes = int(np.size(simsopt_dofs) / 3)
        modes = [int(np.ceil(n/2)) * (-1) ** n for n in range(nmodes)]
        desc_coil = FourierXYZCoil(
            current=current.current,
            X_n=simsopt_dofs[0:nmodes],
            Y_n=simsopt_dofs[nmodes:2 * nmodes],
            Z_n=simsopt_dofs[2 * nmodes:],
            modes=modes
            )
        desc_coils.append(desc_coil)
    desc_coils = CoilSet(desc_coils, NFP=5, sym=True) 

    return desc_coils

def get_x_lines(coils=get_w7x_coils_desc(), 
                NFP=5, 
                R0=5.453448682852216, 
                Z0=0.877797352929447, 
                npts_per_FP=50, 
                basis='xyz'):
    """
    Parameters
    ----------
    coils : CoilSet
    NFP : integer
    R0 : float
        Radial coordinate of x-point at ζ=0
    Z0 : float
        Vertical coordinate of x-point at ζ=0
    npts_per_FP : integer
        Number of points to trace at use per field period
    basis : string
        Either 'xyz' or 'rpz'. The former returns the Cartesian coordinates of 
        the NFP x-lines, while the latter returns the cylindrical coordiantes of
        the x-lines.

    Returns
    -------
    coords : array-like
        Coordinates of the NFP x-lines.

    """

    npts = NFP * npts_per_FP
    zeta = np.linspace(0, 2 * np.pi, npts, endpoint=False)
    r, z = field_line_integrate(R0, Z0, zeta, coils)
    B_xpt = coils.compute_magnetic_field([R0, 0.0, Z0])[0]
    sign_BT = np.sign(B_xpt[1]) 
    zeta *= sign_BT # field lines are traced backwards if B_zeta is negative

    if basis == 'xyz':
        x, y = r * np.cos(zeta), r * np.sin(zeta)
        xs, ys, zs = [], [], []
        for i in range(NFP):
            x_closed = np.concatenate([x, [x[0]]])
            y_closed = np.concatenate([y, [y[0]]])
            z_closed = np.concatenate([z, [z[0]]])
            xs.append(x_closed)
            ys.append(y_closed)
            zs.append(z_closed)
            r = np.roll(r, npts_per_FP)
            z = np.roll(z, npts_per_FP)
            x = r * np.cos(zeta)
            y = r * np.sin(zeta)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        xs, ys, zs = map(np.flip, [xs, ys, zs])
        xs, ys, zs = map(lambda x: np.roll(x, shift=1), (xs, ys, zs))
        return xs, ys, zs
    elif basis == 'rpz':
        rs, ps, zs = [], [], []
        for i in range(NFP):
            r_closed = np.concatenate([r, [r[0]]])
            z_closed = np.concatenate([z, [z[0]]])
            phi_closed = np.concatenate([zeta, [zeta[0]]])
            rs.append(r_closed)
            ps.append(phi_closed)
            zs.append(z_closed)
            r = np.roll(r, npts_per_FP)
            z = np.roll(z, npts_per_FP)
        rs = np.array(rs)
        ps = np.array(ps)
        zs = np.array(zs)
        rs, ps, zs = map(np.flip, [rs, ps, zs])
        rs, ps, zs = map(lambda x: np.roll(x, shift=1), (rs, ps, zs))
        return rs, ps, zs
    raise ValueError('basis must be either "xyz" or "rpz"')
