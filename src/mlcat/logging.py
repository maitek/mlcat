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
        
    def print_metrics(self):
        metrics_str = "".join(
            ["{}: {} | ".format(k,v) for k, v in self.metrics.items()]
        )
        print("| {} ".format(metrics_str))
    

    def plot_history(self,var):
        plt.plot(self.history[var],label=var)
