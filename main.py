import zonjets as zj

atm = zj.Atmosphere(N=128, p = 4, nu = 1e21)


for k in range(100):
    atm.calc(1000)
    print(atm.R_beta)
    atm.plot_zeta(show=False, save=True)
    atm.plot_U(show=False, save=True)
    atm.plot_Ux(show=False, save=True)
    atm.plot_Uy(show=False, save=True)
    # atm.plot_Ek(show=False, save=True)
    # atm.plot_Zk(show=False, save=True)
    # if atm.R_beta > 2:
    #     break

