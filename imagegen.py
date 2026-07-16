#usage: python imagegen.py [image.png] [--out=hahah.pkl]
#forward SAR simulator: bright pixels of a picture become point scatterers, a radar
#platform sweeps a synthetic aperture past them, and the raw range profiles land in a
#pickle that backpro.py can focus back into the picture
import sys
import time
import pickle
import numpy as np
from PIL import Image

C = 299792458.0   #speed of light (m/s)
FC = 4.3e9        #P452 UWB center frequency (Hz)

IMG_PX = 256      #scene raster side (pixels)
SCENE = 40.0      #scene side (m)
STANDOFF = 45.0   #scene center distance from the track (m) - the whole scene sits on one
                  #side so the left/right mirror can't fold the picture onto itself, and
                  #far enough out that range rings sweep the image nearly row by row
APERTURE = 20.0   #synthetic aperture length (m)
PULSES = 1000     #scan points along the aperture
ALT = 10.0        #platform altitude (m)
SAG = 1.5         #drift toward the scene mid-flight (m) - a perfectly straight track can't
                  #tell left from right, so give backpro's mirror tie-break a bend to use
JITTER = 0.25     #random lateral wobble sigma (m) - a smooth bend only smears the mirror
                  #ghost into an arc; random wobble decoheres it outright, so the true side
                  #wins the tie-break by a wide margin instead of a speckle coin flip
NBINS = 3093      #range bins
RMAX = 1000.0     #last range bin (m)
THRESH = 0.02     #reflectivity floor - dimmer pixels are dropped
MARGIN = 0.2      #fraction of the scene left empty around the figure: backpro centers its
                  #frame on the brightest coarse blob, which wanders across the figure, so
                  #the figure must be enough smaller than the frame to survive the wobble
GAMMA = 0.38      #brightness -> reflectivity exponent: flat enough that every part of the
                  #figure clears backpro's 30% range-profile cut (so the frame holds the
                  #whole body), steep enough to keep the shirt/pants tonal blocks apart
NOISE = 1e-3      #receiver noise sigma, relative to signal rms
CHUNK = 64        #pulses simulated per vectorized block

FLAGS = [a for a in sys.argv[1:] if a.startswith("-")]
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]


def image_to_scatterers(path):
    #grayscale composited onto black (transparent background = no echo), padded to a
    #square so the picture isn't stretched, then downsampled; brightness = reflectivity
    im = Image.open(path).convert("RGBA")
    g = Image.composite(im.convert("L"), Image.new("L", im.size, 0), im.getchannel("A"))
    g = g.crop(g.getbbox())   #trim whatever margins the file came with, then add our own
    side = round(max(g.size) / (1 - MARGIN))
    sq = Image.new("L", (side, side), 0)
    sq.paste(g, ((side - g.size[0]) // 2, (side - g.size[1]) // 2))
    refl = (np.asarray(sq.resize((IMG_PX, IMG_PX), Image.LANCZOS), float) / 255.0) ** GAMMA
    #backpro sizes its frame from the range profile (a 30% cut), and a row's echo only grows
    #with sqrt of its width - so narrow rows (head, feet) would get framed out; lift quiet
    #rows toward the loudest row's energy, and cap pixels so 2-px rows can't outshine the
    #whole picture (the display pins its dB scale to the brightest pixel)
    e = np.sqrt((refl ** 2).sum(axis=1))
    refl *= np.clip(0.4 * e.max() / np.maximum(e, 1e-9), 1.0, None)[:, None]
    refl = np.minimum(refl, 1.0)
    row, col = np.nonzero(refl > THRESH)
    step = SCENE / (IMG_PX - 1)
    x = col * step - SCENE / 2               #left edge of the picture -> x = -20
    y = STANDOFF + SCENE / 2 - row * step    #top row -> far edge, so it reconstructs upright
    return np.column_stack([x, y]), refl[row, col]


def flight_track():
    t = np.linspace(-1.0, 1.0, PULSES)
    pos = np.zeros((PULSES, 3))
    pos[:, 0] = t * APERTURE / 2
    pos[:, 1] = SAG * (1 - t * t) + JITTER * np.random.default_rng(1).standard_normal(PULSES)
    pos[:, 2] = ALT
    return pos


def simulate(xy, refl, pos, rbins):
    #each pulse: exact 3d range to every scatterer, reflectivity split between the two
    #straddling bins (linear interp) and carrying the round-trip phase exp(4j pi FC R / C) -
    #that phase history across the aperture is what lets backprojection focus azimuth.
    #per-pulse deposits go through bincount on flattened (pulse, bin) indices, which is
    #much faster than np.add.at for the same scatter-add
    dr = rbins[1] - rbins[0]
    k = 4j * np.pi * FC / C
    taper = np.hanning(len(pos))   #antenna beam roll-off across the aperture; without it
                                   #every bright pixel drags -13 dB sinc streaks through the image
    scan = np.empty((len(pos), NBINS), np.complex64)
    for a in range(0, len(pos), CHUNK):
        p = pos[a:a + CHUNK]
        R = np.sqrt((xy[:, 0] - p[:, :1]) ** 2 + (xy[:, 1] - p[:, 1:2]) ** 2 + p[:, 2:] ** 2)
        assert R.max() < RMAX - dr, "scatterer past the last range bin"
        f, j = np.modf(R / dr)
        v = taper[a:a + len(p), None] * refl * np.exp(k * R)
        j = (j + np.arange(len(p))[:, None] * NBINS).astype(np.int64).ravel()
        j = np.concatenate([j, j + 1])
        v = np.concatenate([(v * (1 - f)).ravel(), (v * f).ravel()])
        n = len(p) * NBINS
        scan[a:a + len(p)] = (np.bincount(j, v.real, n) + 1j * np.bincount(j, v.imag, n)).reshape(len(p), NBINS)
    scan *= np.exp(-k * rbins).astype(np.complex64)   #carrier ride, exactly what backpro strips
    return scan


path = ARGS[0] if ARGS else "180-4071226587.png"
out = next((f.split("=", 1)[1] for f in FLAGS if f.startswith("--out=")), "hahah.pkl")

t0 = time.time()
xy, refl = image_to_scatterers(path)
pos = flight_track()
rbins = np.linspace(0.0, RMAX, NBINS)
print(f"{len(refl)} scatterers from {path}")

scan = simulate(xy, refl, pos, rbins)
if NOISE:
    rng = np.random.default_rng(0)
    sig = np.sqrt((np.abs(scan) ** 2).mean())
    scan += (NOISE * sig * (rng.standard_normal(scan.shape) + 1j * rng.standard_normal(scan.shape))).astype(np.complex64)

with open(out, "wb") as fh:
    pickle.dump({
        "scan_data": scan,
        "platform_pos": pos,
        "range_bins": rbins,
        "sim_params": {
            "c": C, "fc": FC, "image_px": IMG_PX, "scene_extent": SCENE,
            "scene_center": (0.0, STANDOFF), "aperture": APERTURE, "pulses": PULSES,
            "altitude": ALT, "track_sag": SAG, "track_jitter": JITTER, "nbins": NBINS,
            "range_max": RMAX, "threshold": THRESH, "gamma": GAMMA,
            "noise_rel": NOISE, "scatterers": len(refl),
            "source_image": path,
        },
    }, fh)
print(f"wrote {out} ({scan.shape[0]}x{scan.shape[1]} scan) in {time.time() - t0:.1f}s")
