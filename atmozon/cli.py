import typer
import time 

from .classes import Atmosphere, Grid

app = typer.Typer()

@app.command()
def build(steps:int = 1000, N: int = 256):
    atm = Atmosphere()
    grid = Grid(N=N)

    start = time.perf_counter()
    grid.calc(atm, steps=steps)
    end = time.perf_counter()
    print(end-start)



@app.callback(invoke_without_command=True)
def context(ctx: typer.Context):
    """
    CLI running
    """
    if ctx.invoked_subcommand is None:
        print("Running a CLI...")