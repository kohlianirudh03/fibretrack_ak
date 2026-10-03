"""3D structure-tensor orientation field + validation on the twisted phantom.

The fibre axis is the eigenvector of the SMALLEST eigenvalue of
J = G_rho * (grad I)(grad I)^T : intensity varies least along the fibre.
"""
import numpy as np
from scipy import ndimage

def structure_tensor(vol, sigma=1.5, rho=5.0):
    v = np.asarray(vol, np.float32)
    if sigma > 0:
        v = ndimage.gaussian_filter(v, sigma)
    gz, gy, gx = np.gradient(v)
    C = {}
    for nm, a, b in (("zz",gz,gz),("yy",gy,gy),("xx",gx,gx),
                     ("zy",gz,gy),("zx",gz,gx),("yx",gy,gx)):
        C[nm] = ndimage.gaussian_filter(a*b, rho)
    return C

def axes_from_tensor(C, sel):
    """Smallest-eigenvalue eigenvector at the selected voxels. Returns
    axes (N,3) in (z,y,x), and the coherence (linear anisotropy)."""
    n = int(sel.sum())
    J = np.empty((n,3,3), np.float32)
    J[:,0,0]=C["zz"][sel]; J[:,1,1]=C["yy"][sel]; J[:,2,2]=C["xx"][sel]
    J[:,0,1]=J[:,1,0]=C["zy"][sel]
    J[:,0,2]=J[:,2,0]=C["zx"][sel]
    J[:,1,2]=J[:,2,1]=C["yx"][sel]
    w, V = np.linalg.eigh(J)                     # ascending
    ax = V[:,:,0]                                # smallest eigenvalue
    ax = np.where((ax[:,0:1] < 0), -ax, ax)      # fold to +z hemisphere
    with np.errstate(divide="ignore", invalid="ignore"):
        coh = (w[:,2]-w[:,0]) / np.maximum(w[:,2]+w[:,0], 1e-12)
    return ax, coh, w

def angles(ax):
    theta = np.degrees(np.arccos(np.clip(ax[:,0], -1, 1)))
    phi   = np.degrees(np.arctan2(ax[:,1], ax[:,2])) % 360.0
    return theta, phi

if __name__ == "__main__":
    import sys, itertools
    sys.path.insert(0,"scripts")
    from twistphantom import build_twisted
    print("building twisted phantom at the measured PSF ...")
    for psf in (0.0, 3.4):
        vol, tr = build_twisted(size=160, axis_offset_um=250.0, psf_vox=psf,
                                cnr=(15.5 if psf>0 else 0.0), seed=3)
        vox=tr["voxel_um"]; y_ax,x_ax=tr["axis_yx_um"]; k=2*np.pi/tr["L_um"]
        zz,yy,xx = np.mgrid[0:160,0:160,0:160].astype(np.float32)
        Y = yy*vox - y_ax; X = xx*vox - x_ax
        R = np.hypot(Y, X)
        theta_true_field = np.degrees(np.arctan(R*k))
        print("\n--- PSF %.1f vox ---"%psf)
        print("%5s %5s %9s %9s %9s"%("sig","rho","med_err","p90_err","bias"))
        for sig,rho in itertools.product((0.8,1.5,2.5),(3.0,5.0,8.0)):
            C = structure_tensor(vol, sig, rho)
            m = np.zeros(vol.shape, bool)
            m[20:140,20:140,20:140] = True
            ax,coh,w = axes_from_tensor(C, m)
            th,ph = angles(ax)
            tt = theta_true_field[m]
            err = th - tt
            print("%5.1f %5.1f %9.2f %9.2f %9.2f"
                  %(sig,rho,np.median(np.abs(err)),np.percentile(np.abs(err),90),
                    np.median(err)))


# --------------------------------------------------------------- fast path
def block_mean(a, f):
    z, y, x = (np.array(a.shape)//f)*f
    return a[:z, :y, :x].reshape(z//f, f, y//f, f, x//f, f).mean((1, 3, 5))

def structure_tensor_coarse(vol, sigma=2.5, rho=16.0, f=4):
    """Same estimator as structure_tensor, evaluated on an f-times coarser grid.

    Gradients stay at full resolution -- that is where the fibre-scale
    information is. Only the tensor INTEGRATION is coarsened: block-averaging
    by f (a box of width f) before a Gaussian of rho/f differs from a Gaussian
    of rho followed by sampling every f voxels only by the convolution with
    that box, which is negligible while f << rho. Validated in val_coarse.py.
    """
    v = np.asarray(vol, np.float32)
    if sigma > 0:
        v = ndimage.gaussian_filter(v, sigma)
    gz, gy, gx = np.gradient(v)
    del v
    C = {}
    for nm, a, b in (("zz",gz,gz),("yy",gy,gy),("xx",gx,gx),
                     ("zy",gz,gy),("zx",gz,gx),("yx",gy,gx)):
        C[nm] = ndimage.gaussian_filter(block_mean(a*b, f).astype(np.float32),
                                        rho/float(f))
    return C
