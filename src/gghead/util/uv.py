import numpy as np


def gen_tritex(vt: np.ndarray, vi: np.ndarray, vti: np.ndarray, texsize: int):
    """
    Copied from MVP
    Create 3 texture maps containing the vertex indices, texture vertex
    indices, and barycentric coordinates

    Parameters
    ----------
        vt: uv coordinates of texels
        vi: triangle list mapping into vertex positions
        vti: triangle list mapping into texel coordinates
        texsize: Size of the generated maps
    """

    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    ntris = vi.shape[0]

    texu, texv = np.meshgrid(
        (np.arange(texsize) + 0.5) / texsize,
        (np.arange(texsize) + 0.5) / texsize)
    texuv = np.stack((texu, texv), axis=-1)

    vt = vt[vti]

    viim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    vtiim = np.zeros((texsize, texsize, 3), dtype=np.int32)
    baryim = np.zeros((texsize, texsize, 3), dtype=np.float32)

    for i in list(range(ntris))[::-1]:
        bbox = (
            max(0, int(min(vt[i, 0, 0], min(vt[i, 1, 0], vt[i, 2, 0])) * texsize) - 1),
            min(texsize, int(max(vt[i, 0, 0], max(vt[i, 1, 0], vt[i, 2, 0])) * texsize) + 2),
            max(0, int(min(vt[i, 0, 1], min(vt[i, 1, 1], vt[i, 2, 1])) * texsize) - 1),
            min(texsize, int(max(vt[i, 0, 1], max(vt[i, 1, 1], vt[i, 2, 1])) * texsize) + 2))
        v0 = vt[None, None, i, 1, :] - vt[None, None, i, 0, :]
        v1 = vt[None, None, i, 2, :] - vt[None, None, i, 0, :]
        v2 = texuv[bbox[2]:bbox[3], bbox[0]:bbox[1], :] - vt[None, None, i, 0, :]
        d00 = np.sum(v0 * v0, axis=-1)
        d01 = np.sum(v0 * v1, axis=-1)
        d11 = np.sum(v1 * v1, axis=-1)
        d20 = np.sum(v2 * v0, axis=-1)
        d21 = np.sum(v2 * v1, axis=-1)
        denom = d00 * d11 - d01 * d01

        if denom != 0.:
            baryv = (d11 * d20 - d01 * d21) / denom
            baryw = (d00 * d21 - d01 * d20) / denom
            baryu = 1. - baryv - baryw

            baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                np.stack((baryu, baryv, baryw), axis=-1),
                baryim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                np.stack((vi[i, 0], vi[i, 1], vi[i, 2]), axis=-1),
                viim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])
            vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :] = np.where(
                ((baryu >= 0.) & (baryv >= 0.) & (baryw >= 0.))[:, :, None],
                np.stack((vti[i, 0], vti[i, 1], vti[i, 2]), axis=-1),
                vtiim[bbox[2]:bbox[3], bbox[0]:bbox[1], :])

    return viim, vtiim, baryim
