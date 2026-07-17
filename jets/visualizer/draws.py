import matplotlib.pyplot as plt
import cmocean

from .plots import plots

def draw(ds, t, path, key, title=None, show=False, save=True, size=4):
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(size-size/10, size))
    plots[key](ds.isel(t=t), axis)
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(f"{path}/{key}.png", dpi=100)
    if show:
        plt.show()
    plt.close()

def draw_fields(ds, t, path, fields=['q','v','psi'], title=None, show=False, save=True, size=4):
    fig, axes = plt.subplots(nrows=1, ncols=len(fields), figsize=(len(fields)*(size-size/6), size))
    for k in range(len(fields)):
        plots[fields[k]](ds.isel(t=t), axes[k])
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(f"{path}/fields.png", dpi=100)
    if show:
        plt.show()
    plt.close()

def draw_spectra(ds, t, path, spectra=['e', 'z'], title=None, show=False, save=True, size=4):
    fig, axes = plt.subplots(nrows=1, ncols=len(spectra), figsize=(len(spectra)*(size-size/6), size))
    for k in range(len(spectra)):
        plots[spectra[k]](ds.isel(t=t), axes[k])
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(f"{path}/spectra.png", dpi=100)
    if show:
        plt.show()
    plt.close()

def draw_means(ds, t, path, means=['e_mean', 'z_mean'], title=None, show=False, save=True, size=4):
    fig, axes = plt.subplots(nrows=1, ncols=len(means), figsize=(len(means)*(size-size/6), size))
    for k in range(len(means)):
        plots[means[k]](ds.isel(t=range(t+1)), axes[k])
    fig.suptitle(title)
    plt.tight_layout()
    if save:
        plt.savefig(f"{path}/means.png", dpi=100)
    if show:
        plt.show()
    plt.close()

def draw_statistics(ds, t, path, fields=['q','v'], spectra=['z', 'e'], means=['z_mean', 'e_mean'], title=None, show=False, save=True, size=4):
    nrows, height_ratios = 0, []
    if len(fields) != 0:
        nrows += 1
        height_ratios.append(1)
    if len(spectra) != 0:
        nrows += 1
        height_ratios.append(1)
    if len(means) != 0:
        nrows += 1
        height_ratios.append(1)


    fig_master = plt.figure(figsize=(2*size, nrows*size))
    gs = fig_master.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios)
    
    krows = 0
    if len(fields) != 0:
        gs_top = gs[krows].subgridspec(1, len(fields))
        axes_top = [fig_master.add_subplot(gs_top[0, i]) for i in range(len(fields))]
        for k, field in enumerate(fields):
            plots[field](ds.isel(t=t), axes_top[k])
        krows+= 1

    if len(spectra) != 0:    
        gs_bot = gs[krows].subgridspec(1, len(spectra))
        axes_bot = [fig_master.add_subplot(gs_bot[0, i]) for i in range(len(spectra))]
        for k, spec in enumerate(spectra):
            plots[spec](ds.isel(t=t), axes_bot[k])
        krows+= 1

    if len(means) != 0:
        gs_bot = gs[krows].subgridspec(1, len(means))
        axes_bot = [fig_master.add_subplot(gs_bot[0, i]) for i in range(len(means))]
        for k, spec in enumerate(means):
            plots[spec](ds.isel(t=range(t+1)), axes_bot[k])
        krows += 1
    
    if title is not None:
        fig_master.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{path}/atm.png", dpi=100)
    if show:
        plt.show()
    
    plt.close(fig_master)