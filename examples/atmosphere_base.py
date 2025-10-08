"""
Модели атмосфер
"""
import numpy as np
import matplotlib.pyplot as plt

km = 1000 #m
deg = np.pi/180 # rad
hour = 3600 # s

theta = 30 * deg # rad
R0 = 6400 * km # m
T = 24 * hour
Omega = 2 * np.pi / T
g0 = 9.8 # m/s^2
rho0 = 1.225 # kg/m^3
p0 = 101_325 # Pa = kg/(m*s^2)
T0 = 300 #K
R_g = 287 # J/(kg*K) 
f_0 = 2 * Omega * np.sin(theta)

class Atmosphere:
    def __init__(self):
        pass
    def calc(self):
        pass
    def draw(self):
        pass

class Hydrostatic(Atmosphere):
    """
    Решаем самое простое уравнение:
    dp/dz = - rho0 * g0
    """
    def __init__(self, N, z0, zH, p0):
        self.N = N
        self.z = np.linspace(z0, zH, N) * km # m
        self.p = np.zeros(N)
        self.p[0] = p0

    def calc(self):
        for k in range(self.N-1):
            self.p[k+1] = self.p[k] - g0*rho0 * (self.z[k+1] - self.z[k])

    def draw(self):
        fig, ax = plt.subplots()
        ax.plot(self.z/km, self.p/p0)
        plt.xlabel('z, km')
        plt.ylabel('p, atm')
        plt.title('Atmosphere p(z)')
        plt.show()

# atm = Hydrostatic(N=100, z0=0.1, zH=1, p0=p0)
# atm.calc()
# atm.draw()

class TermalHydrostatic(Atmosphere):
    def __init__(self, N, z0, zH, p0, rho0, T0):
        self.N = N
        self.T0 = T0
        self.z = np.linspace(z0, zH, N) * km # m
        self.p = np.zeros(N)
        self.rho = np.zeros(N)
        self.p[0] = p0
        self.rho[0] = rho0

    def calc(self):
        for k in range(self.N-1):
            rho = self.p[k] / (R_g * self.T0)
            g = self.grav(self.z[k])

            self.p[k+1] = self.p[k] - g*rho *  (self.z[k+1] - self.z[k])
            self.rho[k] = rho
    
    def draw(self):
        fig, ax = plt.subplots()
        ax.plot(self.z/km, self.p/p0)
        plt.xlabel('z, km')
        plt.ylabel('p, atm')
        plt.title('Atmosphere p(z)')
        plt.show()

        fig, ax = plt.subplots()
        plt.xlabel('z, km')
        plt.ylabel('rho, rho_atm')
        plt.title('Atmosphere rho(z)')
        ax.plot(self.z/km, self.rho/rho0)
        plt.show()

    def grav(self,z):
        return g0 * (1 + z/R0)

# atm = TermalHydrostatic(N=100, z0=0, zH=100, p0=p0, rho0=rho0, T0=T0)
# atm.calc()
# atm.draw()


class Geostrophic(Atmosphere):
    def __init__(self, N=100, M=100):
        self.N, self.M = N, M
        self.u = np.zeros((N,M))
        self.v = np.zeros((N,M))
        self.U = np.zeros((N,M))
        self.p = np.zeros((N,M))
        self.rho = np.zeros((N,M))
        self.z = np.zeros((N,M))
        self.x = np.zeros((N,M))
        self.y = np.zeros((N,M))

    def calc(self, x0=0, xN=2_000, y0=0, yM=2_000, zK = 100, p0=p0, T0=T0):
        # Координаты карты
        dx = (xN - x0)/(self.N-1) 
        dy = (yM - y0)/(self.M-1)
        xm = (xN - x0)/2
        ym = (yM - y0)/2
        for k in range(self.N):
           self.x[k,:] = k*dx + x0
        for k in range(self.M):
           self.y[:,k] = k*dy + y0

        # Координаты высот
        for i in range(self.N):
            for j in range(self.M):
                x, y = self.x[i,j], self.y[i,j]
                r = np.sqrt((x-xm)**2 + (y-ym)**2)
                z = zK*np.cos(3.14 * r/yM) 
                self.z[i, j] = z if z>=0 else 0

        # Координаты давлений
        for i in range(self.N):
            for j in range(self.M):
                z = self.z[i, j]
                if i == 0 or j == 0 or i==self.N-1 or j == self.M-1: 
                    self.p[i,j] = p0
                    self.rho[i,j] = self.p[i,j] / (R_g * T0)
        for i in range(self.N-2):
            for j in range(self.M-2):                    
                rho = self.p[i,j] / (R_g * T0)
                g = self.grav(self.z[i,j])
                self.p[i+1,j+1] = self.p[i,j] - g*rho *  (self.z[i+1,j+1] - self.z[i,j])
                self.rho[i+1,j+1] = self.p[i+1,j+1] / (R_g * T0)


        for i in range(self.N):
            for j in range(self.M):
                if i == 0 or j == 0 or i==self.N-1 or j == self.M-1: 
                    self.u[i,j] = 0
                    self.v[i,j] = 0
        
        for i in range(self.N-2):
            for j in range(self.M-2):    
                rho = self.rho[i,j]          
                self.u[i,j+1] = - 1/(f_0 * rho) * (self.p[i,j+1]-self.p[i,j])/(self.y[i,j+1] - self.y[i,j])
                self.v[i+1,j] = + 1/(f_0 * rho) * (self.p[i+1,j]-self.p[i,j])/(self.x[i+1,j] - self.x[i,j])
        
        self.U = np.sqrt(self.u**2+self.v**2)

    def grav(self,z):
        return g0 * (1 + z/R0)
    
    def draw(self):
        norm = np.max(self.U)/20
        plt.figure(figsize=(8, 6))
        quiver_plot = plt.quiver(self.x, self.y, self.u/norm, self.v/norm, self.U/norm, 
                         angles='xy', scale_units='xy', scale=1, 
                         cmap='viridis', width=0.005)
        
        plt.colorbar(quiver_plot, label='Модуль вектора U')

        plt.title('Карта направления ветра U')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.show() 

# atm = Geostrophic(N=100, M=100)
# atm.calc()
# atm.draw()

class Euler2D:
    def __init__(self, N=10, M=10, K=10):
        self.N, self.M, self.K = N, M, K
        self.u = np.zeros((N,M,K))
        self.v = np.zeros((N,M,K))
        self.U = np.zeros((N,M,K))
        self.p = np.zeros((N,M,K))
        self.rho = np.zeros((N,M,K))
        self.t = np.zeros((N,M,K))
        self.x = np.zeros((N,M,K))
        self.y = np.zeros((N,M,K))
        self.z = np.zeros((N,M,K))
    
    def calc(self, t0=0, tN=10, x0=0, xM=100, y0=0, yK=100, zS = 100, p0=p0, T0=T0):
        # Координаты карты
        dt = (tN - t0)/(self.N-1)
        dx = (xM - x0)/(self.M-1) 
        dy = (yK - y0)/(self.K-1)
        xm = (xM - x0)/2
        ym = (yK - y0)/2
        for k in range(self.N):
           self.t[k,:,:] = k*dt + t0
        for k in range(self.M):
           self.x[:,k,:] = k*dx + x0
        for k in range(self.K):
           self.y[:,:,k] = k*dy + y0

        # Координаты высот
        for i in range(self.M):
            for j in range(self.K):
                x, y = self.x[0,i,j], self.y[0,i,j]
                r = np.sqrt((x-xm)**2 + (y-ym)**2)
                z = zS*np.cos(3.14 * r/yK) 
                self.z[:,i,j] = z if z>=0 else 0
       
        # Координаты давлений
        for i in range(self.M):
            for j in range(self.K):
                z = self.z[:,i,j]
                if i == 0 or j == 0 or i==self.M-1 or j == self.K-1: 
                    self.p[:,i,j] = p0
                    self.rho[:,i,j] = self.p[:,i,j] / (R_g * T0)
        for i in range(self.M-2):
            for j in range(self.K-2):                    
                rho = self.p[:,i,j] / (R_g * T0)
                g = self.grav(self.z[0,i,j])
                self.p[:,i+1,j+1] = self.p[:,i,j] - g*rho *  (self.z[:,i+1,j+1] - self.z[:,i,j])
                self.rho[:,i+1,j+1] = self.p[:,i+1,j+1] / (R_g * T0)

        for i in range(self.M):
            for j in range(self.K):
                if i == 0 or j == 0 or i==self.M-1 or j == self.K-1: 
                    self.u[:,i,j] = 0
                    self.v[:,i,j] = 0

        for k in range(self.N-2):
            for i in range(self.M-2):
                for j in range(self.K-2):    
                    rho = self.rho[k,i,j]          
                    
                    dpdx = - 1/(rho) * (self.p[k,i+1,j]-self.p[k,i,j])/(self.x[k,i+1,j] - self.x[k,i,j])
                    dpdy = - 1/(rho) * (self.p[k,i,j+1]-self.p[k,i,j])/(self.y[k,i,j+1] - self.y[k,i,j])

                    ...

                    
        self.U = np.sqrt(self.u**2+self.v**2)
    
    def grav(self,z):
        return g0 * (1 + z/R0)
    
# atm = Euler2D(N=2,M=11,K=11)
# atm.calc()
