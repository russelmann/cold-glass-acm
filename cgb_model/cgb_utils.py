"""
(c) 2020 Ruslan Guseinov (ruslan.guseinov@ist.ac.at), IST Austria
         Konstantinos Gavriil, TU Wien
Provided under MIT License

Utilities for MDN
"""
import numpy as np
from scipy.spatial.transform import Rotation


# Frequently used variables
IBZ = [5, 6, 9, 10]
BBZ = [i for i in range(16) if not i in IBZ]
BZ_COLS = [f'K_{i:02}_{j}' for i in range(16) for j in range(3)]
IBZ_COLS = [f'K_{i:02}_{j}' for i in IBZ for j in range(3)]
BBZ_COLS = [f'K_{i:02}_{j}' for i in BBZ for j in range(3)]
FBZ = np.array([[i, i + 1, i + 5, i + 4] for i in range(11) if i % 4 != 3])
MROT90 = [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12] # rotate 4x4 indices matrix 90 degrees
MROT270 = [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3] # rotate 4x4 indices matrix -90 degrees
BEZIER_QUADS = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [4, 5, 9, 8], [5, 6, 10, 9], [6, 7, 11, 10], [8, 9, 13, 12], [9, 10, 14, 13], [10, 11, 15, 14]]
squaredNorm = lambda x: np.dot(x, x)
squaredNormMat = lambda x: (x * x).sum(axis=1)


# parameters
# l1, l2: two quad side lengths
# a: quad angle
# v: displacement vector for point p2, not longer than 1/4 of the smallest edge length
# gamma: rotation of the curve plane from face normal
# theta: opening angles of edge curves
# arguments
# two elements: range between them
# single element a: [-a, a]
DEFAULT_CP_PARAM_RANGES = {
    'l1': [0.15, 0.60],
    'l2': [0.15, 0.60],
    'a': [np.pi/3, 2*np.pi/3],
    'v': [np.inf],
    'gamma': [np.inf],
    'theta': [np.pi / 36],
}
DEFAULT_CP_PARAM_RANGES = {k: np.array(p) for k, p in DEFAULT_CP_PARAM_RANGES.items()}


def check_cp_params(cp_params, prng=None):
    if prng is None:
        prng = DEFAULT_CP_PARAM_RANGES
    for key, val in cp_params.items():
        #print(key, val)
        rng = prng[key]
        if len(rng) == 1:
            rng = rng * np.array([-1, 1])
        if isinstance(val, float) or isinstance(val, int):
            outside = val < rng[0] or rng[1] < val
        else:
            outside = (val < rng[0]).any() or (rng[1] < val).any()
        if outside:
            raise ValueError(f'Value {key} = {val} is outside of range{prng[key]}')
    if cp_params['gamma'].shape != (4,):
        raise ValueError(f"Parameter 'gamma' shape is not (4,)")
    if cp_params['theta'].shape != (4, 2):
        raise ValueError(f"Parameter 'theta' shape is not (4, 2)")
    return True


def random_cp_params(prng=None):
    if prng is None:
        prng = DEFAULT_CP_PARAM_RANGES
    uniform = np.random.uniform
    par = {
        'l1': uniform(*prng['l1']),
        'l2': uniform(*prng['l2']),
        'a': uniform(*prng['a']),
    }
    # Sample v
    phi = uniform(0, 2 * np.pi)
    xi = np.arccos(uniform(-1, 0))
    u = uniform(0, 1)
    par['v'] = gen_sphere_vol(np.min([par['l1'], par['l2']]) / 4, phi, xi, u)
    # Sample gamma
    par['gamma'] = uniform(-np.pi/2, np.pi/2, size=4)
    # Sample theta
    sign = np.random.choice((-1, 1), size=(4, 2))
    par['theta'] = sign * np.arccos(uniform(np.cos(prng['theta'][0]), 1, size=(4, 2)))
    return par


def gen_quad(l1, l2, a, v):
    # p0 at origin
    p = np.zeros((4, 3))
    p[1, 0] = l1
    p[3, 0] = l2 * np.cos(a)
    p[3, 1] = l2 * np.sin(a)
    p[2] = p[1] + p[3] + v
    return p


def proc_input_quad(p):
    if p.shape[0] != 4 or p.shape[-1] != 3:
        raise ValueError(f'p must have first-last dimensions (4, 3), not {p.shape}')
    if len(p.shape) == 2:
        m = 1
    elif len(p.shape) == 3:
        m = p.shape[1]
    else:
        raise ValueError('p must have 2 or 3 dimensions')
    return m


def quad2dist(p):
    # Quad to six squared pairwise distances (broadcast)
    m = proc_input_quad(p)
    d = np.zeros((6, m))
    d[0] = squaredNormMat(p[1] - p[0])
    d[1] = squaredNormMat(p[2] - p[1])
    d[2] = squaredNormMat(p[3] - p[2])
    d[3] = squaredNormMat(p[0] - p[3])
    d[4] = squaredNormMat(p[2] - p[0])
    d[5] = squaredNormMat(p[3] - p[1])
    return d


def quad_adapted_frame(p):
    # Find adapted frame for a quad (broadcast)
    m = proc_input_quad(p)

    # face normal
    b = np.cross(p[2] - p[0], p[3] - p[1])
    b /= np.linalg.norm(b, ord=2, axis=1, keepdims=True)

    # diagonals
    g0 = p[2] - p[0]
    g0 /= np.linalg.norm(g0, ord=2, axis=1, keepdims=True)
    g1 = p[3] - p[1]
    g1 /= np.linalg.norm(g1, ord=2, axis=1, keepdims=True)

    # adapted frames
    F = np.zeros((3, m, 3))
    F[0] = g0 + g1
    F[0] /= np.linalg.norm(F[0], ord=2, axis=1, keepdims=True)
    F[2] = np.cross(g0, g1)
    F[2] /= np.linalg.norm(F[2], ord=2, axis=1, keepdims=True)
    F[1] = np.cross(F[2], F[0])
    F[1] /= np.linalg.norm(F[1], ord=2, axis=1, keepdims=True)
    F = np.swapaxes(F, 0, 1)

    return F, b


def gen_rep(cp_params):
    # Generate boundary representation from high-level parameters
    # Parameters:
    #     cp_param (dict or list(dict)): one parameter set or a list of them
    # Returns:
    #     rep (np.array m x 18): representations
    #     bc (np.array m x 3): barycenters
    #     F (np.array m x 3 x 3): adapted frames
    cp_paramsx = cp_params
    if not isinstance(cp_paramsx, list):
        cp_paramsx = [cp_params]
    m = len(cp_paramsx)
    p = np.zeros((4, m, 3))
    gamma = np.zeros((4, m))
    theta = np.zeros((8, m))
    for i, cp_param in enumerate(cp_paramsx):
        p[:, i] = gen_quad(cp_param['l1'], cp_param['l2'], cp_param['a'], cp_param['v'])
        gamma[:, i] = cp_param['gamma'].flatten()
        theta[:, i] = cp_param['theta'].flatten()

    d = quad2dist(p)
    rep = np.r_[d, gamma, theta].T
    bc = np.mean(p, axis=0)
    F, _ = quad_adapted_frame(p)

    if rep.shape[0] == 1:
        rep = rep[0]
        bc = bc[0]
        F = F[0]
    return rep, bc, F


# points of distance up to R from the origin
def gen_sphere_vol(R, phi, theta, u):
    r = R * np.cbrt(u)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


# project a to plane with normal n
def proj(a, n):
    return a - (np.dot(a, n) / np.dot(n, n)) * n


def gen_tang(e, b, gamma, theta):
    # normalize edge
    e = e / np.linalg.norm(e)
    R = Rotation.from_rotvec(gamma * e).as_matrix()
    b = proj(b, e)
    s = R @ b
    t0 =  e * np.cos(theta[0]) + s * np.sin(theta[0])
    t1 = -e * np.cos(theta[1]) + s * np.sin(theta[1])
    return t0, t1


def ogh_tan_from_tan(p0, p1, t0, t1):
    d = p1 - p0

    t0 /= np.linalg.norm(t0)
    t1 /= np.linalg.norm(t1)

    # negative since d is canonical direction
    a0 =  (6 * np.dot(d, t0) * np.dot(t1, t1) - 3 * np.dot(d, t1) * np.dot(t0, t1)) / (4 * np.dot(t0, t0) * np.dot(t1, t1) - np.dot(t0, t1) ** 2)
    a1 = -(6 * np.dot(d, t1) * np.dot(t0, t0) - 3 * np.dot(d, t0) * np.dot(t0, t1)) / (4 * np.dot(t0, t0) * np.dot(t1, t1) - np.dot(t0, t1) ** 2)

    t0 = a0 * t0
    t1 = a1 * t1

    return t0, t1


def gen_cp(cp_param):
    repx = gen_rep(cp_param)
    cp = rep2cp()
    return cp


def gen_cp_legacy(cp_param):
    # Generate control polygon from high-level parameters (no broadcast)
    # DEPRECATED: use gen_cp instead
    l1 = cp_param['l1']
    l2 = cp_param['l2']
    a = cp_param['a']
    v = cp_param['v']
    gamma = cp_param['gamma']
    theta = cp_param['theta']

    cp = np.zeros((16, 3))

    # gen quad
    p = gen_quad(l1, l2, a, v)
    cp[0] = p[0]
    cp[3] = p[1]
    cp[15] = p[2]
    cp[12] = p[3]

    b = np.cross(cp[15] - cp[0], cp[12] - cp[3])
    b /= np.linalg.norm(b)

    ein = [[1, 2], [7, 11], [14, 13], [8, 4]] # internal edge nodes

    # edges cp0 - cp3, cp3 - cp15, cp15 - cp12, cp12 - cp0
    for i0 in range(4):
        i1 = (i0 + 1) % 4
        t0, t1 = gen_tang(p[i1] - p[i0], b, gamma[i0], theta[i0])
        t0, t1 = ogh_tan_from_tan(p[i0], p[i1], t0, t1)
        cp[ein[i0][0]] = p[i0] + t0 / 3
        cp[ein[i0][1]] = p[i1] + t1 / 3

    # interior control points are generated deterministically
    # according to a simple rule
    cp[ 5] = cp[ 1] + cp[ 4] - cp[ 0]
    cp[ 6] = cp[ 2] + cp[ 7] - cp[ 3]
    cp[ 9] = cp[ 8] + cp[13] - cp[12]
    cp[10] = cp[14] + cp[11] - cp[15]

    return cp


def proc_input_cp(cp):
    cpx = cp.copy()
    if not len(cpx.shape) in [2, 3]:
        raise ValueError('cp must have 2 or 3 dimensions')
    input_dims = len(cpx.shape)
    if input_dims == 2:
        cpx = cpx.reshape((1, 16, 3))
    if cpx.shape[1] != 16:
        raise ValueError('cp must have second to last dimension of length 16')
    if cpx.shape[2] != 3:
        raise ValueError('cp must have last dimension of length 3')
    return cpx, input_dims


def cp2rep(cp, internal=True):
    # Convert control polygons to representations
    # Parameters:
    #    cp (np.array 16 x 3 or m x 16 x 3): control polygon(s)
    #    internal (bool): append to output internal nodes representation
    # Returns:
    #    rep (np.array N or m x N): representation(s), N = 18 for only boundary, N = 30 with internal
    #    bc (m x 3): corner node barycenters
    #    F (m x 3 x 3): adapted frames, None if internal=False
    #    nca (m): true if control polygon m is not canonical
    cpx, input_dims = proc_input_cp(cp)

    # make cp corners canonical
    nca = np.linalg.det(cpx[:, [3, 15, 12]] - cpx[:, [0]]) < 0
    cpx[nca] = cpx[nca][:, MROT90]

    cpx = np.swapaxes(cpx, 0, 1)

    m = cpx.shape[1]
    p = np.copy(cpx[ [0, 1, 2, 3, 7, 11, 15, 14, 13, 12, 8, 4]])

    bc = np.mean(p[[0, 3, 6, 9]], axis=0)
    p -= bc # for numerics?

    # squared distances
    d = np.zeros((6,m))
    d[0] = squaredNormMat(p[3] - p[0])
    d[1] = squaredNormMat(p[6] - p[3])
    d[2] = squaredNormMat(p[9] - p[6])
    d[3] = squaredNormMat(p[0] - p[9])
    d[4] = squaredNormMat(p[6] - p[0])
    d[5] = squaredNormMat(p[9] - p[3])

    # face normal
    b = np.cross(p[6] - p[0], p[9] - p[3])
    b /= np.linalg.norm(b, ord=2, axis=1, keepdims=True)

    # edge directions
    #e = np.zeros((4, 3))
    e = np.zeros((4, m, 3))
    e[0] = p[3] - p[0]
    e[1] = p[6] - p[3]
    e[2] = p[9] - p[6]
    e[3] = p[0] - p[9]
    e /= np.linalg.norm(e, ord=2, axis=2, keepdims=True)

    # compute tangents
    t = np.zeros((8, m, 3))
    t[0] = p[1] - p[0]
    t[1] = p[2] - p[3]
    t[2] = p[4] - p[3]
    t[3] = p[5] - p[6]
    t[4] = p[7] - p[6]
    t[5] = p[8] - p[9]
    t[6] = p[10] - p[9]
    t[7] = p[11] - p[0]
    t /= np.linalg.norm(t, ord=2, axis=2, keepdims=True)

    # compute plane span vectors
    s = np.zeros((4,m,3))
    for i in range(4):
        s[i] = t[2 * i] - (t[2 * i] * e[i]).sum(axis=1)[:,None] * e[i]
        flt = np.isclose(s[i], 0).all(axis=1)
        if flt.any():
            sx = t[2 * i + 1] - (t[2 * i + 1] * e[i]).sum(axis=1)[:,None] * e[i]
            s[i][flt] = sx[flt]
            flt = np.isclose(s[i], 0).all(axis=1)
            if flt.any():
                sx = b - (b * e[i]).sum(axis=1)[:,None] * e[i]
                s[i][flt] = sx[flt]
        sg = np.sign((s[i] * b).sum(axis=1))
        s[i] = s[i] * sg[:,None]
    s /= np.linalg.norm(s, ord=2, axis=2, keepdims=True)

    gamma = np.zeros((4,m))
    for i in range(4):
        tmp = np.cross(e[i], b)
        tmp /= np.linalg.norm(tmp, ord=2, axis=1, keepdims=True)
        gamma[i] = np.arcsin((s[i] * tmp).sum(axis=1))

    theta = np.zeros((8,m))
    for i in range(8):
        theta[i] = np.arcsin((t[i] * s[i // 2]).sum(axis=1))

    rep = np.concatenate((d, gamma, theta))

    # Internal vertices
    F = None
    if internal:
        pin = np.copy(cpx[[5, 6, 10, 9]])
        pin -= bc

        # diagonals
        g0 = p[6] - p[0]
        g0 /= np.linalg.norm(g0, ord=2, axis=1, keepdims=True)
        g1 = p[9] - p[3]
        g1 /= np.linalg.norm(g1, ord=2, axis=1, keepdims=True)

        # adapted frames
        F = np.zeros((3, m, 3))
        F[0] = g0 + g1
        F[0] /= np.linalg.norm(F[0], ord=2, axis=1, keepdims=True)
        F[2] = np.cross(g0, g1)
        F[2] /= np.linalg.norm(F[2], ord=2, axis=1, keepdims=True)
        F[1] = np.cross(F[2], F[0])
        F[1] /= np.linalg.norm(F[1], ord=2, axis=1, keepdims=True)
        F = np.swapaxes(F, 0, 1)

        repin = np.matmul(F, np.swapaxes(pin.T, 0, 1))
        repin = repin.T.reshape(-1, m)
        rep = np.concatenate((rep, repin))

    rep = rep.T
    if input_dims == 2:
        rep = rep.flatten()

    return rep, bc, F, nca


def permute_cp(cp):
    # Permute control polygons to get 4 alternative symmetric representations
    # Parameters:
    #    cp (np.array 16 x 3 or m x 16 x 3): control polygon(s)
    # Returns:
    #    cp (np.array (4 * m) x 16 x 3): control polygons (top m cp remain the same)
    cp, _ = proc_input_cp(cp)

    M = np.zeros((4, 16), dtype=np.int32)
    M[0] = np.arange(16)
    M[1] = M[0][::-1]
    M[2] = np.flip(M[1].reshape(4, 4), 1).flatten()
    M[3] = M[2][::-1]

    #M = M[[0, 1, 3, 2]]

    cp = [cpi[Mi] for cpi in cp for Mi in M]
    cp = np.array(cp)

    return cp


def rep_proc(rep):
    # Canonize representation, does not output 8 copies when quad is planar!
    repx = rep.copy()
    for i in range(4):
        if repx[6 + i] >= np.pi / 2:
            repx[6 + i] -= np.pi
        elif repx[6 + i] < -np.pi / 2:
            repx[6 + i] += np.pi
        else:
            continue
        repx[10 + 2 * i] *= -1
        repx[11 + 2 * i] *= -1
    return repx


def sqdist2quad(d):
    # Convert six squared pairwise distances to a quad in canonical orientation (broadcast)
    # Parameters:
    #    d (np.array m x 6): squared pairwise distances (order: p01, p12, p23, p03, p02, p13)
    # Returns:
    #    p (np.array m x 4 x 3): quads
    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix/423898#423898
    m = d.shape[0]
    D = np.zeros((m, 4, 4))
    D[:,0,1] = d[:,0]
    D[:,1,2] = d[:,1]
    D[:,2,3] = d[:,2]
    D[:,0,3] = d[:,3]
    D[:,0,2] = d[:,4]
    D[:,1,3] = d[:,5]
    D += D.swapaxes(1, 2)
    M = (D[:, 0][:, None, 1:] + D[:, 0][:, 1:, None] - D[:, 1:, 1:]) / 2
    p = np.zeros((m, 4, 3))
    w, v = np.linalg.eig(M) # TODO: use np.linalg.eigh
    w = w.clip(min=0)
    v *= np.sqrt(w)[:, None, :]
    sg = np.sign(np.linalg.det(v))
    np.place(sg, sg==0, [1])
    v *= sg[:, None, None]
    p[:, 1:] = v
    return p


# TODO: slow, not full broadcast
def rep2cp(rep, bc_ext=None, F_ext=None):
    # Convert representation into control polygon
    # Parameters:
    #    rep (np.array m x 18 or m x 30): representations (with or without internal nodes)
    #    bc_ext (np.array m x 3): target barycenter
    #    F_ext (np.array m x 3 x 3): target adapted frame orientation
    # Returns:
    #    cp (np.array m x 16 x 3): control polygons
    repx = rep
    bc_extx = bc_ext
    F_extx = F_ext
    input_dims = len(repx.shape)
    if input_dims == 1:
        repx = rep.reshape(1, -1)
        if not bc_extx is None:
            bc_extx = bc_ext.reshape(1, 3)
        if not F_extx is None:
            F_extx = F_ext.reshape(1, 3, 3)
    m = repx.shape[0]
    cp = np.zeros((16, m, 3))

    d, gamma, theta, repin = np.split(repx, [6, 10, 18], axis=1)
    theta = theta.reshape(-1, 4, 2)
    repin = repin.reshape(-1, 4, 3)

    # compute corner points
    p = sqdist2quad(d)
    p = p.swapaxes(0, 1)
    bc = np.mean(p, axis=0)
    p -= bc
    cp[[0, 3, 15, 12]] = p

    b = np.cross(cp[15] - cp[0], cp[12] - cp[3])
    b /= np.linalg.norm(b, ord=2, axis=1, keepdims=True)

    ein = [[1, 2], [7, 11], [14, 13], [8, 4]] # internal edge nodes

    # edges cp0 - cp3, cp3 - cp15, cp15 - cp12, cp12 - cp0
    # TODO: this can be broadcasted too
    for j in range(m):
        for i0 in range(4):
            i1 = (i0 + 1) % 4
            t0, t1 = gen_tang(p[i1][j] - p[i0][j], b[j], gamma[j][i0], theta[j][i0])
            t0, t1 = ogh_tan_from_tan(p[i0][j], p[i1][j], t0, t1)
            cp[ein[i0][0]][j] = p[i0][j] + t0 / 3
            cp[ein[i0][1]][j] = p[i1][j] + t1 / 3

    # diagonals
    g0 = cp[15] - cp[0]
    g0 /= np.linalg.norm(g0, ord=2, axis=1, keepdims=True)
    g1 = cp[12] - cp[3]
    g1 /= np.linalg.norm(g1, ord=2, axis=1, keepdims=True)

    # adapted frame
    F = np.zeros((3, m, 3))
    F[0] = g0 + g1
    F[0] /= np.linalg.norm(F[0], ord=2, axis=1, keepdims=True)
    F[2] = np.cross(g0, g1)
    F[2] /= np.linalg.norm(F[2], ord=2, axis=1, keepdims=True)
    F[1] = np.cross(F[2], F[0])
    F[1] /= np.linalg.norm(F[1], ord=2, axis=1, keepdims=True)
    F = np.swapaxes(F, 0, 1)

    if repin.size == 0:
        cp[ 5] = cp[ 1] + cp[ 4] - cp[ 0]
        cp[ 6] = cp[ 2] + cp[ 7] - cp[ 3]
        cp[ 9] = cp[ 8] + cp[13] - cp[12]
        cp[10] = cp[14] + cp[11] - cp[15]
    else:
        cp[[5, 6, 10, 9]] = (repin @ np.linalg.inv(F).swapaxes(1, 2)).swapaxes(0, 1)

    if not F_extx is None:
        B = np.linalg.inv(F_extx) @ F
        cp = (cp.swapaxes(0, 1) @ B.swapaxes(1, 2)).swapaxes(0, 1)

    if not bc_extx is None:
        cp = cp + bc_extx

    cp = cp.swapaxes(0, 1)

    if input_dims == 1:
        cp = cp[0]

    return cp
