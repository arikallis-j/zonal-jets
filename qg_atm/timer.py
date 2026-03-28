from time import time

class Timer:
    def __init__(self):
        pass
    def __enter__(self):
        self.time_enter = time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        delta_time = (time() - self.time_enter)
        if delta_time < 10**(-3):
            print(f"{(delta_time/10**(-6)):.2f} μs")
        elif delta_time < 1:
            print(f"{(delta_time/10**(-3)):.2f} ms")
        else:
            print(f"{(delta_time):.2f} s")