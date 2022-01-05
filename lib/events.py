#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import yaml
import scipy.stats
from scipy.integrate import quad
from math import pow, log10, sqrt, sin, pi

from astropy.io import ascii
from astropy.table import Table

import ROOT

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s::%(module)s(l%(lineno)s)::%(funcName)s::%(message)s')


def computeNorm(N, gamma, logEnergy0, energyRange):
    norm = N * (gamma - 1)
    en0 = pow(10, logEnergy0)
    en = [energyRange[0], energyRange[1]]
    norm /= pow(en[0] / en0, -gamma + 1) - pow(en[1] / en0, -gamma + 1)
    return norm


def powerLawIntegral(norm, gamma, logEnergy0, logEnergyRange):
    en0 = pow(10, logEnergy0)
    en = [pow(10, logEnergyRange[0]), pow(10, logEnergyRange[1])]
    integral = norm / (gamma - 1)
    integral *= pow(en[0] / en0, -gamma + 1) - pow(en[1] / en0, -gamma + 1)
    return integral


def binomialUncertainty(n, N):
    if n == 0:
        return 0
    return (1. / N) * sqrt(n * (1 - n / N))


def evalTable(table, logEn, key):
    if key not in ['eff', 'sigma_en', 'delta_en']:
        logging.error('Wrong key for evalTable (eff or sigma_en)')
        raise NameError

    for lge_lo, lge_hi, value in zip(table['lge_lo'], table['lge_hi'], table[key]):
        if (logEn > lge_lo and logEn < lge_hi):
            return value
    return 0


def integrateGaussianPDF(mean, sig, x1, x2):
    def singleArgFunction(x):
        value = scipy.stats.norm.pdf(x, mean, sig)
        return value
    res, _ = quad(singleArgFunction, x1, x2)
    return res


def fileNameFromConfig(config, primary, site, zenith, array='alpha'):
    file_path = config['files']['directory']
    file_path += '/' + config['files'][primary][site][zenith]['directory']
    file_path += '/' + config['files'][primary][site][zenith][array]
    return file_path

########################################
class EventsMC:

    def __init__(
        self,
        nFiles=1,
        primary='gamma',
        site='Paranal',
        zenith=20,
        azimuth=0,
        logEnergyBins=np.linspace(-1.4, 2.0, 50),
        BDTcuts=True,
        test=False,
        nMaxTest=1e2,
        configFile='prod5b_config.yml',
        array='alpha'
    ):
        self.primary = primary
        self.site = site
        self.zenith = zenith
        self.azimuth = azimuth
        self.logEnergy0 = 0
        self.logEnergyBins = logEnergyBins
        self.numberEnergyBins = len(self.logEnergyBins) - 1
        self.BDTcuts = BDTcuts
        self.nFiles = 1 if self.BDTcuts else nFiles
        if self.BDTcuts:
            logging.info('BDTcuts on - nFiles set to 1')
        self.test = test
        self.nMaxTest = nMaxTest
        self.array = array

        self.data = dict()
        self.dataTelescopes = dict()

        self.validateSite()

        # Loading config file
        with open(configFile, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.logEnergyCenter = list()
        for i in range(len(self.logEnergyBins) - 1):
            # energy center
            c = (self.logEnergyBins[i] + self.logEnergyBins[i + 1]) / 2.
            self.logEnergyCenter.append(c)

        self.energyEfficiencyIsLoaded = False
        self.energyResolutionIsLoaded = False

        self.totalSim = self.config['nShowersTotal'][self.site][self.zenith]

        self.specNorm = computeNorm(
            N=self.totalSim,
            gamma=2,
            logEnergy0=self.logEnergy0,
            energyRange=self.config['fullEnergyRange'][self.site][self.zenith]
        )

    def loadTreeData(self):
        logging.info('Loading Tree Data')
        self.data['energy_mc'] = list()
        self.data['energy_rec'] = list()
        self.data['lge_mc'] = list()
        self.data['lge_rec'] = list()
        self.data['nshow'] = dict()

        self.dataTelescopes['energy_mc'] = list()
        self.dataTelescopes['n_sst'] = list()
        self.dataTelescopes['n_mst'] = list()
        self.dataTelescopes['n_lst'] = list()

        def collectEntry(entry):
            self.data['energy_mc'].append(entry.MCe0)
            self.data['energy_rec'].append(entry.erec)
            self.data['lge_mc'].append(log10(entry.MCe0))
            self.data['lge_rec'].append(log10(entry.erec))

        def collectEntryTelescope(entry):
            self.dataTelescopes['energy_mc'].append(entry.MCe0)
            self.dataTelescopes['n_sst'].append(entry.NImages_Ttype[0])
            self.dataTelescopes['n_mst'].append(entry.NImages_Ttype[1])
            self.dataTelescopes['n_lst'].append(entry.NImages_Ttype[2])

        def isEntryValid(entry):
            return not (entry.erec < 0)

        for n in range(self.nFiles):

            fileName = fileNameFromConfig(
                self.config,
                self.primary,
                self.site,
                self.zenith,
                self.array,
            )

            logging.debug('Filename:: {}'.format(fileName))

            file = ROOT.TFile.Open(fileName)
            nSelected = 0
            badEvents = 0
            goodEvents = 0

            for entry in file.DL2EventTree:

                if not isEntryValid(entry):
                    badEvents += 1
                    continue
                goodEvents += 1

                # telescope participation
                # collectEntryTelescope(entry)

                # entry.CutClass is stored as an hex representation of a int
                # We need the function chr to compare it with a int 5
                if entry.CutClass == chr(5) or not self.BDTcuts:
                    collectEntry(entry)
                    nSelected += 1

                if self.test and nSelected > self.nMaxTest:
                    logging.debug('Test=True - Stopping early')
                    break

    def validateSite(self):
        if self.site not in ['Paranal', 'LaPalma']:
            logging.error('Wrong site')
            raise Exception()

    def loadEnergyEfficiency(self):
        logging.info('Loading Energy Efficiency')

        numberTrig = list()
        numberSim = list()
        self.energyEfficiency = list()
        self.energyEfficiencyErr = list()
        self.energyEfficiencyRelErr = list()

        nTrigHist, _ = np.histogram(self.data['lge_mc'], bins=self.logEnergyBins)

        for i in range(len(self.logEnergyCenter)):
            # number trig
            nTrig = nTrigHist[i]
            numberTrig.append(nTrig)

            # number sim
            nSim = powerLawIntegral(
                norm=self.specNorm,
                gamma=2,
                logEnergy0=self.logEnergy0,
                logEnergyRange=[self.logEnergyBins[i], self.logEnergyBins[i + 1]]
            )
            numberSim.append(nSim)

            # efficiency
            eff = nTrig / nSim if nTrig > 3 else 0
            effErr = binomialUncertainty(n=nTrig, N=nSim)
            self.energyEfficiency.append(eff)
            self.energyEfficiencyErr.append(effErr)
            self.energyEfficiencyRelErr.append(effErr / eff if eff > 0 else 0)

        self.energyEfficiencyIsLoaded = True

    def loadEnergyResolution(self):
        logging.info('Loading Energy Resolution')

        self.sigmaEnergy = list()
        self.deltaEnergy = list()
        self.energyDeviation = list()
        self.logEnergyDeviation = list()
        for i in range(len(self.logEnergyCenter)):
            self.energyDeviation.append(list())
            self.logEnergyDeviation.append(list())

        for (r, t) in zip(self.data['lge_rec'], self.data['lge_mc']):
            for i in range(len(self.logEnergyCenter)):
                if t > self.logEnergyBins[i] and t < self.logEnergyBins[i + 1]:
                    self.energyDeviation[i].append((10**r - 10**t) / 10**t)
                    self.logEnergyDeviation[i].append((r - t) / t)
                    break

        self.sigmaEnergy = [np.std(d) for d in self.energyDeviation]
        self.deltaEnergy = [np.mean(d) for d in self.energyDeviation]
        self.energyResolutionIsLoaded = True

    def getNBins(self):
        return len(self.logEnergyCenter)

    def getLogEnergyCenter(self, bin):
        return self.logEnergyCenter[bin]

    def plotEnergyHistogram(self, bin, **kwargs):
        logging.info('Plotting Energy Histograms')
        ax = plt.gca()
        m = self.deltaEnergy[bin]
        s = self.sigmaEnergy[bin]
        histBins = np.linspace(m - 4 * s, m + 4 * s, 50)
        ax.hist(
            x=self.energyDeviation[bin],
            histtype='step',
            bins=histBins,
            **kwargs
        )

    def plotEnergyEfficiency(self, **kwargs):
        logging.info('Plotting Energy Efficiency')

        if not self.energyEfficiencyIsLoaded:
            self.loadEnergyEfficiency()

        ax = plt.gca()
        ax.errorbar(
            self.logEnergyCenter,
            self.energyEfficiency,
            yerr=self.energyEfficiencyErr,
            **kwargs
        )

    def plotEnergyResolution(self, **kwargs):
        logging.info('Plotting Energy Resolution')

        if not self.energyResolutionIsLoaded:
            self.loadEnergyResolution()

        ax = plt.gca()
        ax.plot(self.logEnergyCenter, self.sigmaEnergy, **kwargs)

    def plotEnergyBias(self, **kwargs):
        logging.info('Plotting Energy Bias')

        if not self.energyResolutionIsLoaded:
            self.loadEnergyResolution()

        ax = plt.gca()
        ax.plot(self.logEnergyCenter, self.deltaEnergy, **kwargs)

    def plotEnergyEfficiencyRelErr(self, **kwargs):
        logging.info('Plotting Energy Efficiency RelErr')

        if not self.energyEfficiencyIsLoaded:
            self.loadEnergyEfficiency()

        ax = plt.gca()
        ax.plot(self.logEnergyCenter, self.energyEfficiencyRelErr, **kwargs)

    def plotRadiusEfficiency(self, energyRange, **kwargs):
        logging.info('Plotting Radius Efficiency')

        scatR = self.config['scatRadius'][self.site][self.zenith]
        radiusBins = np.linspace(0, scatR, 50)
        # radiusBins = np.linspace(0, 200, 40)
        numberTrig = list()
        numberSim = list()
        radiusCenter = list()
        radiusEfficiency = list()

        radiusHist = [
            r for (r, e) in zip(self.data['radius_mc'], self.data['energy_mc'])
            if (e > energyRange[0]) and (e < energyRange[1])
        ]
        nTrigHist, _ = np.histogram(radiusHist, bins=radiusBins)

        for i in range(len(radiusBins) - 1):
            # radius center
            c = (radiusBins[i] + radiusBins[i + 1]) / 2.
            radiusCenter.append(c)

            # number trig
            nTrig = nTrigHist[i]
            numberTrig.append(nTrig)

            # number sim
            fracInRadius = (radiusBins[i + 1]**2 - radiusBins[i]) / scatR**2
            nSimInEnergy = powerLawIntegral(
                norm=self.specNorm,
                gamma=2,
                logEnergy0=self.logEnergy0,
                logEnergyRange=[log10(energyRange[0]), log10(energyRange[1])]
            )
            nSim = nSimInEnergy * fracInRadius
            numberSim.append(nSim)

            # efficiency
            radiusEfficiency.append(nTrig / nSim if nTrig > 3 else 0)

        ax = plt.gca()
        ax.plot(radiusCenter, radiusEfficiency, **kwargs)

    def plotThetaEfficiency(self, energyRange, **kwargs):
        logging.info('Plotting Theta Efficiency')

        thetaBins = np.linspace(0, self.config['viewCone'], 30)
        numberTrig = list()
        numberSim = list()
        thetaCenter = list()
        thetaEfficiency = list()

        thetaHist = [
            t for (t, e) in zip(self.data['theta_mc'], self.data['energy_mc'])
            if (e > energyRange[0]) and (e < energyRange[1])
        ]
        nTrigHist, _ = np.histogram(thetaHist, bins=thetaBins)

        for i in range(len(thetaBins) - 1):
            # theta center
            c = (thetaBins[i] + thetaBins[i + 1]) / 2.
            thetaCenter.append(c)

            # number trig
            nTrig = nTrigHist[i]
            numberTrig.append(nTrig)

            # number sim
            fracInTheta = (
                sin(thetaBins[i + 1] * pi / 360)**2 - sin(thetaBins[i] * pi / 360)**2
            ) / sin(self.config['viewCone'] * pi / 360)**2

            nSimInEnergy = powerLawIntegral(
                norm=self.specNorm,
                gamma=2,
                logEnergy0=self.logEnergy0,
                logEnergyRange=[log10(energyRange[0]), log10(energyRange[1])]
            )
            nSim = nSimInEnergy * fracInTheta
            numberSim.append(nSim)

            # efficiency
            thetaEfficiency.append(nTrig / nSim if nTrig > 3 else 0)

        ax = plt.gca()
        ax.plot(thetaCenter, thetaEfficiency, **kwargs)

    def plotMeanNshow(self, withHistogram=False, **kwargs):

        nShowAll = list()
        for i in range(len(self.logEnergyCenter)):
            nShowAll.append(list())

        def findEnergyBin(en):
            lge = log10(en)
            for i in range(len(self.logEnergyCenter)):
                if lge > self.logEnergyBins[i] and lge < self.logEnergyBins[i + 1]:
                    return i
            return None

        def collectNshow(en, evs):
            # GLOBAL nShowAll
            i = findEnergyBin(en)
            if i is not None:
                nShowAll[i].append(len(evs))

        for en, evs in self.data['nshow'].items():
            collectNshow(en=en, evs=evs)

        self.meanNshow = list()
        self.histNshow = list()
        nshowBins = np.linspace(0.5, 10.5, 11)
        for i in range(len(self.logEnergyCenter)):
            if len(nShowAll[i]) > 0:
                self.meanNshow.append(np.mean(nShowAll[i]))
                h, b = np.histogram(nShowAll[i], nshowBins)
                self.histNshow.append(h)
            # else:
            #     self.meanNshow.append(0)
            #     self.histNshow.append(np.zeros(10))

        for i in range(len(self.histNshow)):
            if np.sum(self.histNshow[i]) > 0:
                hist = self.histNshow[i] / np.sum(self.histNshow[i])
                self.histNshow[i] = hist

        ax = plt.gca()

        if withHistogram:
            X, Y = np.meshgrid(self.logEnergyBins, np.linspace(0.5, 10.5, 11))
            C = np.array([list(h) for h in self.histNshow])
            C = C.T

            cm = ax.pcolormesh(X, Y, C, cmap='ocean_r', edgecolors='None', linewidth=0.2)
            cbar = plt.colorbar(cm, format='%.1f%%')
            cbar.ax.get_yaxis().labelpad = 15
            # cbar.ax.get_yaxis().labelsize = 18
            # cbar.ax.get_yaxis().titlesize = 18
            cbar.ax.set_ylabel('fraction', rotation=270)

        ax.plot(self.logEnergyCenter, self.meanNshow, **kwargs)

    def plotHistNshow(self, energyBin, **kwargs):

        nShowAll = list()
        for en, evs in self.data['nshow'].items():
            lge = log10(en)
            if lge > self.logEnergyBins[energyBin] and lge < self.logEnergyBins[energyBin + 1]:
                nShowAll.append(len(evs))

        ax = plt.gca()
        ax.hist(
            nShowAll,
            density=True,
            bins=np.linspace(0.5, 10.5, 11),
            **kwargs
        )

    def plotNtelescopes(self, **kwargs):

        minLgE = log10(min(self.dataTelescopes['energy_mc']))
        maxLgE = log10(max(self.dataTelescopes['energy_mc']))

        for tel in ['n_sst', 'n_mst', 'n_lst']:
            sns.regplot(
                x=[log10(e) for e in self.dataTelescopes['energy_mc']],
                y=self.dataTelescopes[tel],
                x_bins=self.logEnergyBins,
                fit_reg=None,
                **kwargs
            )

    def exportTable(self):
        if not self.energyEfficiencyIsLoaded:
            self.loadEnergyEfficiency()
        if not self.energyResolutionIsLoaded:
            self.loadEnergyResolution()

        tableData = dict()
        tableData['lge'] = self.logEnergyCenter

        binsToWriteHigh, binsToWriteLow = list(), list()
        for i in range(len(self.logEnergyCenter)):
            binsToWriteLow.append(self.logEnergyBins[i])
            binsToWriteHigh.append(self.logEnergyBins[i + 1])

        tableData['lge_lo'] = binsToWriteLow
        tableData['lge_hi'] = binsToWriteHigh

        tableData['eff'] = self.energyEfficiency
        tableData['eff_err'] = self.energyEfficiencyErr

        tableData['sigma_en'] = self.sigmaEnergy
        tableData['delta_en'] = self.deltaEnergy

        outName = (
            'data/eff_' + self.primary
            + '_' + self.site + '_' + self.array
            + '_z' + str(self.zenith) + '_az'
            + str(self.azimuth) + '.cvs'
        )
        ascii.write(Table(tableData), outName, format='basic', overwrite=True)


########################################
class EffectiveArea:
    def __init__(
        self,
        N,
        index,
        logEnergyBins=np.linspace(-1.6, 2.0, int((2.0 + 1.6) * 5)),
        primary='gamma',
        site='Paranal',
        zenith=20,
        azimuth=0,
        color='k',
        marker='o',
        label='None',
        useRecEnergy=True,
        useBias=True,
        configFile='prod5b_config.yml'
    ):
        self.size = N
        self.index = index
        self.logEnergyBins = logEnergyBins
        self.primary = primary
        self.site = site
        self.zenith = zenith
        self.azimuth = azimuth
        self.color = color
        self.marker = marker
        self.label = label
        self.useRecEnergy = useRecEnergy
        self.useBias = useBias

        # Loading config file
        with open(configFile, 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.computeEffArea()

    def computeEffArea(self):
        fileName = (
            'data/eff_' + self.primary
            + '_' + self.site + '_' + self.array
            + '_z' + str(self.zenith) + '_az'
            + str(self.azimuth) + '.cvs'
        )

        table = dict(ascii.read(fileName, format='basic'))

        norm = computeNorm(
            N=self.size,
            gamma=self.index,
            logEnergy0=0,
            energyRange=self.config['fullEnergyRange'][self.site][self.zenith]
        )
        logging.warning('Energy range hardcoded')
        self.effArea = list()
        self.effAreaErr = list()
        self.logEnergy = list()
        totArea = (self.config['scatRadius'][self.site][self.zenith]**2) * pi
        logging.warning('Scaterring radius hardcoded')

        self.numberSim = [0] * (len(self.logEnergyBins) - 1)
        self.numberRec = [0] * (len(self.logEnergyBins) - 1)

        for iBin in range(len(self.logEnergyBins) - 1):
            logEn0 = self.logEnergyBins[iBin]
            logEn1 = self.logEnergyBins[iBin + 1]

            c = (logEn0 + logEn1) / 2.
            self.logEnergy.append(c)

            logEnSteps = np.linspace(logEn0, logEn1, 11)

            for iStep in range(len(logEnSteps) - 1):
                logEnStepCenter = (logEnSteps[iStep] + logEnSteps[iStep + 1]) / 2
                nSim = powerLawIntegral(
                    norm=norm,
                    gamma=self.index,
                    logEnergy0=0,
                    logEnergyRange=[logEnSteps[iStep], logEnSteps[iStep + 1]]
                )
                eff = evalTable(
                    table=table,
                    logEn=logEnStepCenter,
                    key='eff'
                )
                nTrig = eff * nSim

                self.numberSim[iBin] += nSim
                if not self.useRecEnergy:
                    self.numberRec[iBin] += nTrig
                else:
                    sigRel = evalTable(
                        table=table,
                        logEn=logEnStepCenter,
                        key='sigma_en'
                    )
                    deltaRel = evalTable(
                        table=table,
                        logEn=logEnStepCenter,
                        key='delta_en'
                    )
                    mean = (
                        (1 + deltaRel) * (10**logEnStepCenter)
                        if self.useBias else 10**logEnStepCenter
                    )
                    sig = sigRel * (10**logEnStepCenter)
                    for jBin in range(len(self.logEnergyBins) - 1):
                        minE = 10**self.logEnergyBins[jBin]
                        maxE = 10**self.logEnergyBins[jBin + 1]
                        if maxE > mean - 4 * sig and minE < mean + 4 * sig:
                            fractionInBin = integrateGaussianPDF(
                                mean=mean,
                                sig=sig,
                                x1=minE,
                                x2=maxE
                            )
                            self.numberRec[jBin] += fractionInBin * nTrig

        self.effArea = [
            (totArea * n / N if N > 0 else 0) for (n, N) in zip(self.numberRec, self.numberSim)
        ]
        self.effAreaErr = [
            (totArea * binomialUncertainty(n=n, N=N) if N > 0 else 0)
            for (n, N) in zip(self.numberRec, self.numberSim)
        ]

    def plotEffArea(self, **kwargs):
        ax = plt.gca()
        ax.errorbar(
            self.logEnergy,
            self.effArea,
            yerr=self.effAreaErr,
            color=self.color,
            linestyle='None',
            marker=self.marker,
            label=self.label,
            **kwargs
        )

    def plotNumberRec(self, **kwargs):
        ax = plt.gca()
        ax.plot(
            self.logEnergy,
            self.numberRec,
            color=self.color,
            marker=self.marker,
            label=self.label,
            **kwargs
        )

    def plotEffAreaRelErr(self, **kwargs):
        ax = plt.gca()
        ax.plot(
            self.logEnergy,
            [r / e for (e, r) in zip(self.effArea, self.effAreaErr)],
            color=self.color,
            marker=self.marker,
            label=self.label,
            **kwargs
        )

    def maxUncertainty(self):
        return np.amax([r / e for (e, r) in zip(self.effArea, self.effAreaErr)])
