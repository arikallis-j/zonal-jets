def time_R_beta(ds, k):
    return f"t = {ds.isel(t=k)['t']:.1f} | $R_{'{'}\\beta{'}'}$ = {ds.isel(t=k)['R_beta']:.1f}"

def time_title(ds, k):
    return f"t = {ds.isel(t=k)['t']:.1f}"

titles = {
    't': time_title,
    't-R_b': time_R_beta, 
}