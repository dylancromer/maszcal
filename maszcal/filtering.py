import numpy as np
from scipy.interpolate import interp1d
import pixell.enmap
import pixell.utils
import pixell.fft


FROM_ARCMIN = 2 * np.pi / 360 / 60
TO_ARCMIN = 1/FROM_ARCMIN


def filter_kappa(kappa_interpolator, thetas, kmask):
    modrmap = kmask.modrmap()
    kappa_2d = pixell.enmap.enmap(kappa_interpolator(modrmap), kmask.wcs)
    filtered_kappa_2d = filter_map(kappa_2d, kmask)
    centers = thetas*TO_ARCMIN
    delta_theta = np.diff(centers).mean()/2
    bin_edges = np.concatenate((centers[0:1]-delta_theta, centers+delta_theta))
    binner = Bin2D(modrmap, bin_edges * pixell.utils.arcmin)
    filtered_centers, kappa_filtered_and_binned = binner.bin(kappa_2d)
    if not np.allclose(filtered_centers/pixell.utils.arcmin, centers):
        raise RuntimeError('Something went wrong: could not reconstruct thetas from binner')
    return kappa_filtered_and_binned


class FilterBinner(object):
    def __init__(self, data_root):
        self.kmask = pixell.enmap.read_map(f"{data_root}_kmask.fits")
        self.modrmap = self.kmask.modrmap()
        bin_edges = np.loadtxt(f"{data_root}_bin_edges.txt")
        self.binner = bin2D(self.modrmap, bin_edges*pixell.utils.arcmin)

    def get_binned(self,thetas,profile):
        th_kappa_2d = pixell.enmap.pixell.enmap(
            interp(thetas,profile)(self.modrmap),
            self.kmask.wcs,
        ) # attach WCS to interpolated array
        filt_kappa_2d = filter_map(th_kappa_2d, self.kmask)
        fcents, th_binned = self.binner.bin(filt_kappa_2d)
        return fcents, th_binned


def interp(x, y, bounds_error=False, fill_value=0., **kwargs):
    return interp1d(x, y, bounds_error=bounds_error, fill_value=fill_value, **kwargs)


def filter_map(imap,kfilter):
    return pixell.enmap.enmap(
        np.real(
            pixell.fft.ifft(
                pixell.fft.fft(
                    imap,
                    axes=[-2, -1],
                )*kfilter,
                axes=[-2, -1],
                normalize=True,
            ),
        ),
        imap.wcs,
    )


class Bin2D(object):
    def __init__(self, modrmap, bin_edges):
        self.centers = (bin_edges[1:]+bin_edges[:-1])/2
        self.digitized = np.digitize(modrmap.reshape(-1), bin_edges, right=True)
        self.bin_edges = bin_edges
        self.modrmap = modrmap

    def bin(self, data2d, weights=None, err=False, get_count=False, mask_nan=False):
        if weights is None:
            if mask_nan:
                keep = ~np.isnan(data2d.reshape(-1))
            else:
                keep = np.ones((data2d.size, ), dtype=bool)
            count = np.bincount(self.digitized[keep])[1:-1]
            res = np.bincount(self.digitized[keep], (data2d).reshape(-1)[keep])[1:-1]/count
            if err:
                meanmap = self.modrmap.copy().reshape(-1) * 0
                for i in range(self.centers.size):
                    meanmap[self.digitized==i] = res[i]
                std = np.sqrt(
                    np.bincount(
                        self.digitized[keep],
                        (
                            (data2d-meanmap.reshape(self.modrmap.shape))**2
                        ).reshape(-1)[keep])[1:-1]/(count-1)/count,
                )
        else:
            count = np.bincount(self.digitized, weights.reshape(-1))[1:-1]
            res = np.bincount(self.digitized, (data2d*weights).reshape(-1))[1:-1]/count
        if get_count:
            assert not(err) # need to make more general
            return self.centers, res, count
        if err:
            assert not(get_count)
            return self.centers, res, std
        return self.centers, res
