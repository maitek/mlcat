import matplotlib.pyplot as plt
import csv
import json
import numpy as np

class MetricWatcher():
    def __init__(self, watch_vars):
        self.watch_vars = watch_vars
        self.history = [] # list of historic metrics
        self.last_metric = {}

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

    def update(self, scope):
        
        metric = {}
        for var in self.watch_vars:
            metric[var] = scope.get(var, "")
            
        self.history.append(metric)
        self.last_metric = metric
        
        
    def print_metrics(self, metrics=None):
        if metrics is None:
            metrics = self.watch_vars
        
        metrics_str = "".join(
            ["{}: {} | ".format(k,v) for k, v in self.last_metric.items() if k in metrics]
        )
        print("| {} ".format(metrics_str))
    

    def plot_metrics(self, metrics=None, show=True):
        if metrics is None:
            metrics = self.watch_vars
        for var in metrics:
            data = [x.get(var,np.nan) for x in self.history]
            plt.plot(data,label=var)
        if show:
            plt.legend()
            plt.show()

    def export_csv(self, file_name):
        with open(file_name, 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, self.history[0].keys())
            w.writeheader()
            for metric in self.history:
                w.writerow(metric)

    def export_json(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.history,f)

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



