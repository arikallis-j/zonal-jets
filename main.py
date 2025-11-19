import zonjets as zj

atm = zj.Atmosphere(p = 4, nu = 1e21, epsilon=1e-13)
print(atm.epsilon)


while True:
    atm.calc(1000)
    print(atm.R_beta)
    atm.plot_zeta(show=False, save=True)
    atm.plot_U(show=False, save=True)
    if atm.R_beta > 3:
        break

