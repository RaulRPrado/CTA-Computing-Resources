#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse

from lib.events import EffectiveArea

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
        '-m',
        '--mode',
        help=(
            'Mode: size (compares different number of events),'
            'bins (compare different binning) or rec (compares rec and true energy)'
        ),
        type=str,
        default='size',
        choices=['size', 'bins', 'rec']
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
        '-z',
        '--zenith',
        help='Zenith: 20, 40 or 60',
        type=int,
        default=20,
        choices=[20, 40, 60]
    )
    parser.add_argument(
        '-i',
        '--index',
        help='Spectral index',
        type=float,
        default=2.0
    )
    parser.add_argument(
        '-n',
        '--size',
        help='List of log size',
        type=float,
        nargs='+',
        default=[8, 8.5, 9]
    )
    parser.add_argument('--show', help='Show plots', action='store_true')
    args = parser.parse_args()

    show = args.show

    zenith = args.zenith
    index = args.index

    logELim = {20: [-1.8, 2.4], 40: [-1.6, 2.8], 60: [-1.2, 2.8]}
    logging.warning('logELim hardcoded')

    # Reporting input parameters
    logging.info('Running eff_area with the following input parameters:')
    logging.info('Site (-s): {}'.format(args.site))
    logging.info('Array (-s): {}'.format(args.array))
    logging.info('Mode (-m): {}'.format(args.mode))
    logging.info('Spectral index (-i): {}'.format(args.index))
    logging.info('Size (-n): {}'.format(args.size))
    logging.info('Zenith angle (-z): {}'.format(args.zenith))

    def getLogEnergyBins(zenith, nBins):
        return np.linspace(
            logELim[zenith][0],
            logELim[zenith][1],
            int((logELim[zenith][1] - logELim[zenith][0]) * nBins + 1)
        )

    if args.mode == 'size':
        dataSize = [10**n for n in args.size]
        colors = ['k', 'r', 'b', 'g']
        colors = colors[0:len(dataSize)]
        markers = ['o', 's', '^', '<']
        markers = markers[0:len(dataSize)]
        effArea = list()
        binsPerDecade = 5
        for s, c, m in zip(dataSize, colors, markers):
            effArea.append(EffectiveArea(
                N=s,
                index=index,
                zenith=zenith,
                site=args.site,
                array=args.array,
                logEnergyBins=getLogEnergyBins(zenith, binsPerDecade),
                color=c,
                marker=m,
                label='log10(N)={:.2}'.format(log10(s)),
                useRecEnergy=True
            ))
    elif args.mode == 'bins':
        dataSize = 3e7 if index == 1.5 else 3e8
        colors = ['k', 'r']
        markers = ['o', 's']
        binsPerDecade = [5, 10]
        effArea = list()
        for n, c, m in zip(binsPerDecade, colors, markers):
            effArea.append(EffectiveArea(
                N=dataSize,
                index=index,
                zenith=zenith,
                site=args.site,
                array=args.array,
                logEnergyBins=getLogEnergyBins(zenith, n),
                color=c,
                marker=m,
                label='{} bins/decade'.format(n),
                useRecEnergy=True
            ))
    elif args.mode == 'rec':
        dataSize = 3e7 if index == 1.5 else 3e8
        colors = ['k', 'r']
        markers = ['o', 's']
        rec = [True, False]
        effArea = list()
        for r, c, m in zip(rec, colors, markers):
            effArea.append(EffectiveArea(
                N=dataSize,
                index=index,
                zenith=zenith,
                site=args.site,
                array=args.array,
                logEnergyBins=getLogEnergyBins(zenith, 5),
                color=c,
                marker=m,
                label=r'$E_\mathrm{R}$' if r else r'$E_\mathrm{T}$',
                useRecEnergy=r
            ))
    else:
        logging.error('Wrong mode')
        raise ValueError()

    # Relative Uncertainties
    fig = plt.figure(figsize=(8, 6), tight_layout=True)

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xlabel(r'log$_{10}$($E$/TeV)')
    ax.set_ylabel(r'$\sigma(A_\mathrm{eff})/A_\mathrm{eff}$')

    for e in effArea:
        e.plotEffAreaRelErr()

    ax.legend(frameon=False)
    ax.set_ylim(0.001, 0.5)
    xlim = [-1.8, 2.8]
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()
    ax.plot(xlim, [0.02, 0.02], linestyle=':', color='k')
    ax.plot([log10(0.04), log10(0.04)], ylim, linestyle='--', color='k')
    ax.plot([2, 2], ylim, linestyle='--', color='k')

    ax.autoscale(tight=True, axis='both')

    if args.show:
        plt.show()
    else:
        figName = (
            'figures/EffAreaRelErr' + str(args.site)
            + '_' + str(args.array)
            + '_z' + str(zenith)
            + '_index' + str(index) + '_' + args.mode
        )
        logging.info('Printing figures {}.pdf/png'.format(figName))
        plt.savefig(figName + '.png', format='png', bbox_inches='tight')
        plt.savefig(figName + '.pdf', format='pdf', bbox_inches='tight')

    # Effective Area
    fig = plt.figure(figsize=(8, 6), tight_layout=True)

    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xlabel(r'log$_{10}$($E$/TeV)')
    ax.set_ylabel(r'$A_\mathrm{eff}$ [m$^2$]')

    for e in effArea:
        e.plotEffArea()

    ax.legend(frameon=False)
    xlim = [-1.8, 2.8]
    ax.set_xlim(xlim)

    # ax.set_ylim(0.001, 0.5)

    # xlim = ax.get_xlim()
    # ax.plot(xlim, [0.02, 0.02], linestyle=':', color='k')

    # ax.autoscale(tight=True, axis='x')

    if args.show:
        plt.show()
    else:
        figName = (
            'figures/EffArea' + str(args.site)
            + '_' + str(args.array)
            + '_z' + str(zenith)
            + '_index' + str(index) + '_' + args.mode
        )
        logging.info('Printing figures {}.pdf/png'.format(figName))
        # binsLabel = '_bins' if args.mode == 'comparingBins' else ''
        plt.savefig(figName + '.png', format='png', bbox_inches='tight')
        plt.savefig(figName + '.pdf', format='pdf', bbox_inches='tight')
