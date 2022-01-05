#!/usr/bin/python3

import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_pdf import PdfPages

from lib.events import EventsMC

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s::%(module)s(l%(lineno)s)::%(funcName)s::%(message)s')


labelsize = 20
plt.rc('font', family='serif', size=labelsize)
plt.rc('xtick', labelsize=labelsize)
plt.rc('ytick', labelsize=labelsize)
plt.rc('text', usetex=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-z',
        '--zenith',
        help='Zenith: 20, 40 or 60  Default: [20, 40, 60]',
        type=float,
        nargs='+',
        default=[20, 40, 60]
    )
    parser.add_argument(
        '-s',
        '--site',
        help='Site: Paranal or LaPalma - Default: Paranal',
        type=str,
        default='Paranal',
        choices=['Paranal', 'LaPalma']
    )
    parser.add_argument(
        '-a',
        '--array',
        help='Array: alpha, alpha-SSTs, alpha-MSTs or alpha-LSTs',
        type=str,
        default='alpha',
        choices=['alpha', 'alpha-SSTs', 'alpha-MSTs', 'alpha-LSTs']
    )
    parser.add_argument(
        '-t',
        '--test',
        help='Turns on test mode where only a small fraction of the events will be used',
        action='store_true'
    )
    args = parser.parse_args()

    colors = ['k', 'r', 'b', 'k']
    markers = ['o', '>', '^', 's']
    lge_min = {20: -1.8, 40: -1.6, 60: -1.2}
    lge_max = {20: 2.4, 40: 2.8, 60: 2.8}

    events = list()
    for z in args.zenith:
        range0 = np.linspace(lge_min[z], 1.0, int((1.0 - lge_min[z]) * 20))
        range0 = np.delete(range0, [len(range0) - 1])
        range1 = np.linspace(1.0, lge_max[z], int((lge_max[z] - 1.0) * 10))
        logEnergyBins = (np.concatenate((range0, range1), axis=0))

        ev = EventsMC(
            nFiles=1,
            primary='gamma',
            logEnergyBins=logEnergyBins,
            test=args.test,
            zenith=z,
            BDTcuts=True,
            site=args.site,
            nMaxTest=1e5,
            array=args.array,
        )
        ev.loadTreeData()

        ev.loadEnergyResolution()
        ev.loadEnergyEfficiency()
        ev.exportTable()
        events.append(ev)

    ################
    # Efficiency
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    # ax.set_title('diffuse gamma')
    ax.set_xlabel(r'log$_{10}$($E$/TeV)')
    ax.set_ylabel(r'efficiency, $\epsilon$')

    for ev, c, m in zip(events, colors, markers):
        ev.plotEnergyEfficiency(
            color=c,
            linestyle='--',
            marker=m,
            markersize=4,
            label=r'$\theta$ = ' + str(ev.zenith)
        )

    ax.legend(frameon=False)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1)

    figName = 'figures/Efficiency' + args.site + '_' + args.array
    logging.info('Saving figure: {}'.format(figName))
    plt.savefig(figName + '.png', format='png', bbox_inches='tight')
    plt.savefig(figName + '.pdf', format='pdf', bbox_inches='tight')

    ################
    # Efficiency RelErr
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    # ax.set_title('diffuse gamma')
    ax.set_xlabel(r'log$_{10}$($E$/TeV)')
    ax.set_ylabel(r'$\sigma(\epsilon)/\epsilon$')

    for ev, c, m in zip(events, colors, markers):
        ev.plotEnergyEfficiencyRelErr(
            color=c,
            linestyle='--',
            marker=m,
            markersize=4,
            label=r'$\theta$ = ' + str(ev.zenith)
        )
    ax.legend(frameon=False)
    ax.set_ylim(0.001, 0.1)

    figName = 'figures/EfficiencyRelErr' + args.site + '_' + args.array
    logging.info('Saving figure: {}'.format(figName))
    plt.savefig(figName + '.png', format='png', bbox_inches='tight')
    plt.savefig(figName + '.pdf', format='pdf', bbox_inches='tight')

    # Not plotting resolutions by now
    exit()

    ################
    # Energy Resolution
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel(r'log$_{10}$($E$/TeV)')
    ax.set_ylabel(r'$\sigma(E_\mathrm{R})/E_\mathrm{T}$')

    for ev, c, m in zip(events, colors, markers):
        ev.plotEnergyResolution(
            color=c,
            linestyle='--',
            marker=m,
            markersize=4,
            label=r'$\theta$ = ' + str(ev.zenith)
        )
    ax.legend(frameon=False)
    ylim = ax.get_ylim()
    ax.set_ylim(0, ylim[1])

    if show:
        plt.show()
    else:
        plt.savefig('figures/EnergyResolution' + site + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/EnergyResolution' + site + '.pdf', format='pdf', bbox_inches='tight')

    ################
    # Energy Bias
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel(r'log$_{10}$($E$/TeV)')
    ax.set_ylabel(r'$\delta E_\mathrm{R}/E_\mathrm{T}$')

    for ev, thr, c, m in zip(events, threshold, colors, markers):
        ev.plotEnergyBias(
            color=c,
            linestyle='--',
            marker=m,
            markersize=4,
            label=r'$\theta$ = ' + str(ev.zenith) + thresholdLabel(thr)
        )
    ax.legend(frameon=False)

    if show:
        plt.show()
    else:
        plt.savefig('figures/EnergyBias' + site + '.png', format='png', bbox_inches='tight')
        plt.savefig('figures/EnergyBias' + site + '.pdf', format='pdf', bbox_inches='tight')

    ################
    # Energy Deviation Histograms
    for ev in events:
        nBins = ev.getNBins()

        for iBin in range(nBins):
            fig = plt.figure(iBin + 4, figsize=(8, 6), tight_layout=False)
            plt.clf()

            ax = plt.gca()
            ax.set_title(
                r'log$_{10}$($E$/TeV) = ' + '{:.2f}'.format(ev.getLogEnergyCenter(iBin))
            )
            ax.set_xlabel(r'$(E_\mathrm{R}-E_\mathrm{T})/E_\mathrm{T}$')

            ev.plotEnergyHistogram(bin=iBin, color='k')
            ylim = ax.get_ylim()
            d = ev.deltaEnergy[iBin]
            ax.plot([d, d], ylim, linestyle=':', color='k')
            ax.autoscale(tight=True, axis='y')

        with PdfPages('figures/EnergyHistograms' + site + '_z' + str(ev.zenith) + '.pdf') as pdf:
            for iBin in range(nBins):
                fig = plt.figure(iBin + 4)
                pdf.savefig(fig)
