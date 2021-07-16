import numpy as np
import matplotlib.pyplot as plt

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

textstr = '\n'.join((
    r'$V=30$',
    r'$SC=9$',
    r'$R_{cell}$=3',
    r'pathloss_exp = 3.5',
    r'$P_{cap}=100$',
    r'$\gamma=0.05$',
    r'SC cache cap = 3'))

lru_result = (871.91 + 962.18 + 969.39 + 960.68 + 908.50) / 5
lfu_result = (709.38 + 814.44 + 821.93 + 880.46 + 807.98) / 5
lmin_result = (279.97 + 317.47 + 280.60 + 275.54 + 310.58) / 5


fig, ax = plt.subplots()
ax.bar([0, 1, 2], [lru_result, lfu_result, lmin_result], tick_label=['LRU','LFU', 'LMIN'])
ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
ax.set_title("Avg power consumption, averaged over the last 5 time slots")
plt.show()