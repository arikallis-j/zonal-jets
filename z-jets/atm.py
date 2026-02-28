from abc import ABC, abstractmethod
import numpy as np
import os

class PluginManager:
    def __init__(self):
        self._registry = {}
        self._usage = {}

    def register(self, port_type):
        def decorator(implementation):
            self._registry.setdefault(port_type, []).append(implementation)
            return implementation
        return decorator
    
    def setup(self, config):
        for (port_type, name) in config.items():
            for impl in self._registry[port_type]:
                if impl.name == name:
                    self._usage[port_type] = impl

    def get(self, name):
        return self._usage[name]
    
    def get_all(self):
        return self._registry
    
plugin_manager = PluginManager()

class StatePort(ABC):
    @abstractmethod
    def __init__(self, field, t):
        pass

class IntegratorPort(ABC):
    @abstractmethod
    def __init__(self, delta):
        pass
    @abstractmethod
    def step(self, state, n_iter):
        pass

class DiagnosticsPort(ABC):
    @abstractmethod
    def analyse(self, state):
        pass

class IOManagerPort(ABC):
    @abstractmethod
    def write(self, data):
        pass

class DriverPort(ABC):
    @abstractmethod
    def pipeline(self, n_iter):
        pass

@plugin_manager.register("state")
class SimpleState(StatePort):
    name = "simple-state"
    def __init__(self, field = np.zeros((10,10)), t=0):
        self.field = field
        self.t = t

@plugin_manager.register("integrator")
class SimpleIntegrator(IntegratorPort):
    name = "simple-integrator"
    def __init__(self, delta = 1):
        self.delta = delta

    def step(self, state, n_iter=1):
        for _ in range(n_iter):
            state.field += 1
            state.t += self.delta
        return state

@plugin_manager.register("diagnostics")
class SimpleDiagnostics(DiagnosticsPort):
    name = "simple-diagnostics"
    def __init__(self):
        self.regime = "test"
    def analyse(self, state):
        cur_time = state.t
        mean_field = np.mean(state.field)
        data = cur_time, mean_field
        print(f"time = {cur_time} | mean-field = {mean_field}")
        return data

@plugin_manager.register("io-manager")
class SimpleIOManager(IOManagerPort):
    name = "simple-io-manager"
    def __init__(self):
        self.regime = "test"
    def write(self, data):
        cur_time, mean_field = data
        if not os.path.exists("data"):
            os.mkdir('data')
        with open('data/data.txt', 'w') as f:
            data = f"time = {cur_time} | mean-field = {mean_field}"
            f.write(data)

@plugin_manager.register("driver")
class SimpleDriver(DriverPort):
    name = "simple-driver"
    def __init__(self):
        regime = "test"
    def pipeline(self, n_iter, state, integrator, diagnostics, io_manager):
        for _ in range(n_iter):
            state = integrator.step(state, 10)
            data = diagnostics.analyse(state)
            io_manager.write(data)
        return state

class Simulation:
    def __init__(self, config):
        plugin_manager.setup(config)
        self.state = plugin_manager.get("state")()
        self.integrator = plugin_manager.get("integrator")()
        self.diagnostics = plugin_manager.get("diagnostics")()
        self.io_manager = plugin_manager.get("io-manager")()
        self.driver = plugin_manager.get("driver")()

    def run(self, n_iter=1):
        self.driver.pipeline(n_iter, 
                             self.state, 
                             self.integrator, 
                             self.diagnostics, 
                             self.io_manager)

config = {
    'state': 'simple-state',
    'integrator': 'simple-integrator',
    'diagnostics': 'simple-diagnostics',
    'io-manager': 'simple-io-manager',
    'driver': 'simple-driver',
}

model = Simulation(config)
model.run()
