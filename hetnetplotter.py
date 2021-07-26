import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import pickle
import argparse

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
line_styles = ['-', '--', '-.', ':']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process output files.')
    parser.add_argument('--filenames', type=str, nargs='+',
                        help='pickled file to be processed')
    parser.add_argument('--mode', type=str, default='barcomp',
                        help='Plotting mode (bar comparison, sinr constraint trend, cache cap trend etc.',
                        choices=['barcomp', 'sinr', 'cache'])

    myargs = parser.parse_args()
    res = {}
    args = []
    s = []

    for filename in myargs.filenames:
        print("Processing...")
        try:
            with open(filename, 'rb') as f:
                args, wirelessStats = pickle.load(f)
        except (IOError, OSError) as e:
            print(e)
            continue

        if args.hetnet_load is not None:
            s = args.hetnet_load.split("_")
            s = [re.findall('\d.*', x) for x in s]
            s2 = []
            for l in s[1:5]:
                for i in l:
                    s2.append(float(i))
            s = s2
        else:
            s = [args.graph_size, *args.hetnet_params]

        print(wirelessStats['w_by_slot'])
        print(wirelessStats['w_by_slot'][-1])
        if myargs.mode=='barcomp':
            res[args.cache_type] = np.mean(wirelessStats['power_by_slot'][-6:-1])
        elif myargs.mode=='sinr':
            if args.cache_type not in list(res.keys()):
                res[args.cache_type] = {}
            res[args.cache_type][args.wireless_consts[1]] = np.mean(wirelessStats['power_by_slot'][-6:-1])
        elif myargs.mode=='cache':
            if args.cache_type not in list(res.keys()):
                res[args.cache_type] = {}
            res[args.cache_type][args.max_capacity] = np.mean(wirelessStats['power_by_slot'][-6:-1])

    print(res)
    cache_types = list(res.keys())
    textstr = '\n'.join((
        "$V=%d$" %(s[0]),
        "$SC=%d$" %(s[1]),
        "$R_{cell}=%.2f$" %(s[2]),
        "Pathloss exp = %.2f" %(s[3]),
        "Catalog size = %d" %(args.catalog_size),
        "Demand size = %d" %(args.demand_size)))

    if myargs.mode=='barcomp':
        textstr += '\n'.join((
        "\n$P_{cap}=%.2f$" %(args.wireless_consts[0]),
        "$\gamma=%.2f$" %(args.wireless_consts[1]),
        "SC cache cap = %d" %(args.max_capacity)))

        fig = plt.figure(figsize=(3*(len(res)+1), 3*1.0))
        ax = fig.add_subplot(1,1,1)
        ax.bar(range(len(res)), list(res.values()), tick_label=cache_types)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_title("Avg power consumption, averaged over the last 5 time slots")

    elif myargs.mode=='sinr':
        textstr += '\n'.join((
        "\n$P_{cap}=%.2f$" %(args.wireless_consts[0]),
        "SC cache cap = %d" %(args.max_capacity)))

        fig = plt.figure(figsize=(3*(len(res)+1), 3*1.0))
        ax = fig.add_subplot(1,1,1)
        for i, key in enumerate(res.keys()):
            ax.plot(list(res[key].keys()), list(res[key].values()), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], label=key)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_title("Avg power consumption with different sinr, averaged over the last 5 time slots")
        ax.legend(loc='upper center', fontsize='x-large')
    
    elif myargs.mode=='cache':
        textstr += '\n'.join((
        "\n$P_{cap}=%.2f$" %(args.wireless_consts[0]),
        "$\gamma=%.2f$" %(args.wireless_consts[1])))

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for i, key in enumerate(res.keys()):
            print(list(res[key].values()))
            ax.plot(list(res[key].keys()), list(res[key].values()), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], label=key)
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_title("Avg power consumption with different sinr, averaged over the last 5 time slots")
        ax.legend(loc='upper center', fontsize='x-large')


    """fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [res['LRU'], res['LFU'], res['LMIN']], tick_label=['LRU','LFU', 'LMIN'])
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.set_title("Avg power consumption, averaged over the last 5 time slots")"""
    plt.show()