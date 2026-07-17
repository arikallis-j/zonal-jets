from .draws import *
from .titles import titles
from tqdm import tqdm

class Visualizer:
    def __init__(self, path=None, fields=[], s_fields=[], spectra=[], means=[]):
        if path is not None:
            self.setup(path, fields, s_fields, spectra, means)
    
    def setup(self, path, fields=[], s_fields=[], spectra=[], means=[], title='t'):
        self.imag_path = path
        self.fields = fields
        self.s_fields = s_fields
        self.spectra = spectra
        self.means = means
        self.make_title = titles[title]

    def visual(self, ds, k_iter=None, show=False, save=True):
        if k_iter is None:
            k_iter = len(ds['t']) - 1
        title = self.make_title(ds, k_iter)
        draw_statistics(ds, k_iter, self.imag_path, fields=self.fields, spectra=self.spectra, means=self.means, title=title, show=show, save=save)
        
    def visuals(self, climate, show=False, save=True):
        for k in tqdm(range(len(climate['t']))):
            self.visual(climate, k, show=show, save=save)