from jax import jit

from .grid import make_grid
from .model import models
from .initial import initials
from .forcing import forcings

class Atmosphere:
    def __init__(self, model="qgc-1", n_grid=None, params={}):
        self.model = models[model]
        self.meta = self.model['meta']
        self.make_params = self.model['params']
        self.make_atm_phys = self.model['atm_phys']
        self.make_atm_stat = self.model['atm_stat']
        self.make_atm_nums = self.model['atm_nums']
        self.make_atm_dataset = self.model['atm_dataset']
        self.make_init_state = self.model['init_state']
        self.model_rhs = self.model['rhs']

        if n_grid is not None:
            self.setup(n_grid, params)

    def setup(self, n_grid, params={}):
        self.grid = make_grid(n_grid)
        self.params = self.make_params(**params)
        self.rhs = jit(lambda time, field: self.model_rhs(time, field, self.params, self.grid))
        self._physics = jit(lambda time, field: self.make_atm_phys(time, field, self.params, self.grid))
        self._statistics = jit(lambda time, field: self.make_atm_stat(time, field, self.params, self.grid))
        self._numbers = jit(lambda time, field: self.make_atm_nums(time, field, self.params, self.grid))
        self._datasets = lambda time, field, descript={}: self.make_atm_dataset(time, field, self._physics, self._statistics, self._numbers, descript)
    
    def start(self):
        time, field = self.make_init_state(self.params, self.grid)
        return time, field

    def physics(self):
        return self._physics

    def statistics(self):
        return self._statistics

    def numbers(self):
        return self._numbers

    def datasets(self):
        return self._datasets

    def initials(self):
        return list(initials.keys())

    def forcings(self):
        return list(forcings.keys())

    def parameters(self):
        return list(self.make_params()._asdict().keys())

    def description(self):
        return self.meta