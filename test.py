
import diffractix as dfx
from diffractix.elements import Space
from diffractix.beams import GaussianBeam

if __name__ == "__main__":
    beam = GaussianBeam.from_waist(w0=1e-3, wavelength=1e-6, z_waist_loc=1, n = 1.5)
    beam.plot()
    print(beam)