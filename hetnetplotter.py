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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process output files.')
    parser.add_argument('--filenames', type=str, nargs='+',
                        help='pickled file to be processed')

    myargs = parser.parse_args()
    res = {}

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

        print(s)
        textstr = '\n'.join((
            "$V=%d$" %(s[0]),
            "$SC=%d$" %(s[1]),
            "$R_{cell}=%.2f$" %(s[2]),
            "Pathloss exp = %.2f" %(s[3]),
            "$P_{cap}=%.2f$" %(args.wireless_consts[0]),
            "$\gamma=%.2f$" %(args.wireless_consts[1]),
            "SC cache cap = %d" %(args.max_capacity)))

        print(wirelessStats['power_by_slot'])
        res[args.cache_type] = np.mean(wirelessStats['power_by_slot'][-6:-1])

    print(res)
    fig, ax = plt.subplots()
    ax.bar([0, 1, 2], [res['LRU'], res['LFU'], res['LMIN']], tick_label=['LRU','LFU', 'LMIN'])
    ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.set_title("Avg power consumption, averaged over the last 5 time slots")
    plt.show()