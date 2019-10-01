import matplotlib.pyplot as plt

class MetricWatcher():
    def __init__(self, watch_vars):
        self.watch_vars = watch_vars
        self.history = {x : [] for x in watch_vars}
        self.metrics = {}

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

    def update(self, scope):
        
        for var in self.watch_vars:
            self.metrics[var] = scope.get(var, "")
            self.history[var].append(self.metrics[var])
        title = ""
        
    def print_metrics(self, metrics=None):
        if metrics is None:
            metrics = self.watch_vars
        
        metrics_str = "".join(
            ["{}: {} | ".format(k,v) for k, v in self.metrics.items() if k in metrics]
        )
        print("| {} ".format(metrics_str))
    

    def plot_metrics(self, metrics=None, show=True):
        if metrics is None:
            metrics = self.watch_vars
        for var in metrics:
            plt.plot(self.history[var],label=var)
        if show:
            plt.legend()
            plt.show()

## Tests

def test_print_metrics_1():
    with MetricWatcher(watch_vars = ["a", "b"]) as mw:
        for a in range(5):
            b = a**2
            mw.update(scope=locals())
            mw.print_metrics()

def test_print_metrics_2():
    with MetricWatcher(watch_vars = ["a", "b"]) as mw:
        for a in range(5):
            b = a**2
            mw.update(scope=locals())
            mw.print_metrics(metrics=["a"])

## Tests

def test_plot_metrics_1():
    with MetricWatcher(watch_vars = ["a", "b"]) as mw:
        for a in range(5):
            b = a**2
            mw.update(scope=locals())
            mw.plot_metrics()

def test_plot_metrics_2():
    with MetricWatcher(watch_vars = ["a", "b"]) as mw:
        for a in range(5):
            b = a**2
            mw.update(scope=locals())
            mw.plot_metrics(metrics=["a"])



