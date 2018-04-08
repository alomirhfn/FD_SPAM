import time


class Timer:
    def __init__(self):
        self.start_time = 0

    def tic(self):
        self.start_time = time.time()

    def toc(self, task=""):
        print(task + ' took %0.3f s' % (time.time() - self.start_time))
        self.start_time = 0
