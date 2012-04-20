from __future__ import division

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats
import pymc as pm

samples = np.random.randn(1000)

def plot_synth_samples(data, lower_samples=10, upper_samples=200, lower_dps=10, upper_dps=200, interval=50):
    x, y = np.meshgrid(np.arange(lower_samples, upper_samples, interval), np.arange(lower_dps, upper_dps, interval))
    #plt.figure()
    for i in range(len(x)):
        for j in range(len(y)):
            plt.subplot(x.shape[0], x.shape[1], i*x.shape[0]+j+1)
            if i == len(x)-1:
                plt.xlabel('%d'%x[i,j])
            if j == 0:
                plt.ylabel('%d'%y[i,j])
            #plot_synth_like(data, samples=np.round(x[i, j]), dataset_samples=np.round(y[i, j]))

def plot_synth_like(data, lower_mu=-1, upper_mu=1, lower_sig=.8, upper_sig=1.5, steps=50, samples=100, dataset_samples=100, plot=True):
    x, y = np.meshgrid(np.linspace(lower_mu, upper_mu, steps), np.linspace(lower_sig, upper_sig, steps))
    z = np.empty_like(x)
    for i in range(len(x)):
        for j in range(len(y)):
            z[i,j] = synth_likelihood(data, x[i,j], y[i,j], samples=samples, dataset_samples=dataset_samples)

    plt.contourf(x, y, z)

def plot_erroranalysis(data_samples=(10,), dataset_samples=(10,), datasets=200):
    x = dataset_samples
    y = np.empty(x.shape, dtype=np.float)

    for data_sample in data_samples:
        data = np.random.randn(data_sample)

        for i, dataset_sample in enumerate(dataset_samples):
            errors = []
            sl_sum = 0
            pt_sum = 0
            for rep in range(1, 400):
                # Chose two random mu pts
                mu1 = 0
                mu2 = (np.random.rand()-.5)

                # Evaluate true likelihood
                pt1 = pm.normal_like(data, mu=mu1, tau=1**-2)
                pt2 = pm.normal_like(data, mu=mu2, tau=1**-2)

                ptr = pt1 / pt2
                pt_sum += pt1
                pt_sum += pt2

            #print ptr

                # Evaluate synth likelihood
                ps1 = synth_likelihood(data, mu1, 1, dataset_samples=x[i], samples=datasets)
                ps2 = synth_likelihood(data, mu2, 1, dataset_samples=x[i], samples=datasets)

                sl_sum += ps1
                sl_sum += ps2

                pts = ps1 / ps2
            #print pts

                errors.append((pts - ptr)**2)
            print pt_sum
            print sl_sum
            y[i] = np.mean(errors)

        plt.plot(x, y, label='%i' % data_sample)

    plt.xlabel('Number of samples per dataset')
    plt.ylabel('MSE')
    plt.legend()


def plot_ratio_analysis(data_samples=(100,), dataset_samples=(100,), datasets=100):
    x, y = np.meshgrid(data_samples, dataset_samples)
    z = np.empty(x.shape, dtype=np.float)

    for i, data_sample in enumerate(data_samples):
        for j, dataset_sample in enumerate(dataset_samples):
            data = np.random.randn(x[j, i])
            errors = []
            sl_sum = 0
            pt_sum = 0
            for rep in range(1, 200):
                # Chose two random mu pts
                mu1 = (np.random.rand()-.5) * 3
                mu2 = (np.random.rand()-.5) * 3

                # Evaluate true likelihood
                pt1 = pm.normal_like(data, mu=mu1, tau=1)
                pt2 = pm.normal_like(data, mu=mu2, tau=1)

                ptr = pt1 / pt2
                pt_sum += pt1
                pt_sum += pt2

                #print ptr

                # Evaluate synth likelihood
                ps1 = synth_likelihood(data, mu1, 1, dataset_samples=y[j, i], samples=datasets)
                ps2 = synth_likelihood(data, mu2, 1, dataset_samples=y[j, i], samples=datasets)

                sl_sum += ps1
                sl_sum += ps2

                pts = ps1 / ps2
                #print pts

                errors.append((pts - ptr)**2)
            print pt_sum
            print sl_sum
            z[j, i] = np.mean(errors)
            print x[j, i], y[j,i], z[j, i]

    print x
    print y
    print z
    cont = plt.contourf(x, y, z)

    plt.colorbar(cont)
    plt.xlabel('Number of samples per dataset')
    plt.ylabel('Size of input data.')

def synth_likelihood(data, mu, std, samples=100, dataset_samples=100):
    """Compute synthetic likelihood for collapsing threshold wfpt."""
    def mv_normal_like(s, mu, cov):
        s = np.asmatrix(s)
        mu = np.asmatrix(mu)
        cov = np.asmatrix(cov)
        return .5 * (s - mu) * (cov**-1) * (s - mu).T - .5 * np.log(np.linalg.det(cov))

    true_sum = np.array((data.mean(), data.std())) #, np.sum(data), data.var()))

    sum_stats = np.empty((samples, 2))
    for sample in range(samples):
        s = np.random.randn(dataset_samples)*std + mu
        sum_stats[sample,:] = s.mean(), s.std() #, np.sum(s), s.var()

    mean = np.mean(sum_stats, axis=0)
    cov = np.cov(sum_stats.T)
    # Evaluate synth likelihood
    logp = mv_normal_like(true_sum, mean, cov)
    return -logp


synth = pm.stochastic_from_dist(name="Wiener synthetic likelihood",
                                logp=synth_likelihood,
                                dtype=np.float32,
                                mv=False)

mu = pm.Uniform('mu', lower=-5, upper=5, value=0)
std = pm.Uniform('std', lower=.1, upper=2, value=1)

sl = synth('Synthetic likelihood', value=samples, mu=mu, std=std, observed=True)
#rl = pm.Normal('Regular likelihood', value=samples, mu=mu, tau=std**-2, observed=True)