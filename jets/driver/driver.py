from ..integrator import Integrator
from ..atmosphere import Atmosphere
from ..io_manager import IOManager
from ..visualizer import Visualizer

from tqdm import tqdm
from IPython.display import clear_output

class Driver:
    def __init__(self):
        self.iom = IOManager()
        self.vis = Visualizer(self.iom)
        self.atm = Atmosphere()
        self.int = Integrator()

    def setup(self, exp_name='test', visual={'fields': ['q', 'v']}, atmosphere={'n_grid': 256}, integrator={'n_points': 100}):
        self.iom.setup(exp_name=exp_name)
        self.vis.setup(path=self.iom.imag_path, **visual)
        self.atm.setup(**atmosphere)
        self.int.setup(rhs=self.atm.rhs, **integrator)
       
        self.iom.dump_descript(self.atm.meta, self.atm.params, self.int.status)
        self.integral = self.int.integral()
        self.datasets = self.atm.datasets()

    def descript(self):
        descript = self.iom.load_descript(self.iom.exp_name)
        text = ""
        for name, part in descript.items():
            text += f"[{name}]\n"
            for key, val in part.items():
                text += f"{key} = {val}\n"
            text += "\n"
        return text

    def start(self, load=False, save=True, show=False):
        if load:
            self.state = self.int.make_state(*self.iom.load_state())
            self.climate = self.iom.load_dataset()
        else:
            self.state = self.int.make_state(*self.atm.start())
            self.climate = self.iom.state_dataset(self.state, self.datasets)

        self.t_iter = 0

    def run(self, n_iter, load=False, save=True, show=False, logtime=False, tau=10, metrics=None):
        stop, data = False, None
        for k in tqdm(range(n_iter)):
            clear_output(wait=True)
            if k==0:
                self.save(save, show)
                continue
            if ((self.t_iter-1)%(tau-1)==0 and self.t_iter>1) and logtime:
                self.update_timescale(tau=tau)
            self.state = self.integral(self.state)
            self.climate = self.iom.full_dataset([self.climate, self.iom.state_dataset(self.state, self.datasets)])
            self.save(save, show)
            self.t_iter += 1
            if metrics is not None:
                stop, data = metrics(self.climate)
            if stop:
                break
        return data

    def save(self, save=True, show=False):
        self.vis.visual(self.climate, save=save, show=show)
        self.iom.dump_state(self.state)
        self.iom.dump_dataset(self.climate)

    def update_timescale(self, tau=10):
        integrator = self.int.status
        integrator['interval'] *= tau
        integrator['n_points'] *= tau
        self.int.setup(rhs=self.atm.rhs, **integrator)
        self.integral = self.int.integral()