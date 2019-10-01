import matplotlib.pyplot as plt

class MetricWatcher():
    def __init__(self, watch_vars):
        self.watch_vars = watch_vars
        self.history = {x : [] for x in watch_vars}
        #print(self.history)

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

    def update(self):
        global_vars = globals()
        metrics = {}
        
        for var in self.watch_vars:
            metrics[var] = global_vars.get(var, "")
            self.history[var].append(metrics[var])
        title = ""
        
        metrics_str = "".join(["{}: {} | ".format(k,v) for k, v in metrics.items()])
        print("{} | {}".format(title, metrics_str))
    
    def plot_history(self,var):
        plt.plot(self.history[var],label=var)
