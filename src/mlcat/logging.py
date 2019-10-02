import matplotlib.pyplot as plt
import csv
import json

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
            data = [x[var] for x in self.history]
            plt.plot(data,label=var)
        if show:
            plt.legend()
            plt.show()

    # def export_csv():

    #     for 

    #     csv_dict = {"test": 1, "testing": 2}

    #     with open('mycsvfile.csv', 'wb') as f:  # Just use 'w' mode in 3.x
    #         w = csv.DictWriter(f, my_dict.keys())
    #         w.writeheader()
    #         w.writerow(my_dict)

    # def export_json():

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



