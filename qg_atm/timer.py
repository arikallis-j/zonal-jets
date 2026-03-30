from time import time

class Timer:
    def __init__(self):
        self.current_time = time()
    def __enter__(self):
        self.time_enter = time()
    def __exit__(self, exc_type, exc_val, exc_tb):
        delta_time = (time() - self.time_enter)
        print(self.pretty_time(delta_time))
    def check(self):
        period = (time() - self.current_time)
        self.current_time = time()
        return period
    def pretty_time(self, delta_time):
        if delta_time < 10**(-3):
            return f"{(delta_time/10**(-6)):>6.2f} μs"
        elif delta_time < 1:
            return f"{(delta_time/10**(-3)):>6.2f} ms"
        else:
            return f"{(delta_time):>6.2f}  s"
    def standart_time(self, delta_time):
        hours = int(delta_time//3600)
        minutes = int((delta_time%3600)//60)
        seconds = int(delta_time - hours*3600 - minutes*60)
        standart_delta = f"{minutes:02d}:{seconds:02d}"
        if hours!=0:
            standart_delta = f"{hours}:" + standart_delta
        return standart_delta