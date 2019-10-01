
# ML-Cat

Utilities and helpers for machine learning projects
	
~~~~
        /\_/\
_______/ o o \______
                              
~~~~

## Installation

~~~~
pip install git+https://github.com/maitek/mlcat
~~~~


### MetricWatcher

MetricWatcher can watch/log certain variables/metrics during training process.

Example:
~~~~
from mlcat.logging import MetricWatcher

with MetricWatcher(watch_vars = ["a", "b"]) as mw:
    for a in range(5):
        b = a**2
        mw.update(scope=locals())
        mw.print_metrics()
~~~~
~~~~
| a: 0 | b: 0 |  
| a: 1 | b: 1 |  
| a: 2 | b: 4 |  
| a: 3 | b: 9 |  
| a: 4 | b: 16 |  
~~~~

[Notebook example](https://github.com/maitek/mlcat/blob/master/examples/metric_watcher_example.ipynb)
