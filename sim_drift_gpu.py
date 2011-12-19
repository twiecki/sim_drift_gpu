from __future__ import division

import pycuda.compiler
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.curandom
from pycuda.cumath import exp as pycuda_exp
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import scipy.stats
import pymc as pm
from pymc.distributions import mv_normal_like

code = """
    #include <curand_kernel.h>

    extern "C"
    {
    __global__ void sim_drift(curandState *global_state, float const v, float const V, float const a, float const z, float const Z, float const t, float const T, float const dt, float const intra_sv, float *out)
    {
        float start_delay, start_point, drift_rate, rand, prob_up, position, step_size, time;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        curandState local_state = global_state[idx];

        /* Sample variability parameters. */
        start_delay = curand_uniform(&local_state)*T + (t-T/2);
        start_point = (curand_uniform(&local_state)*Z + (z-Z/2))*a;
        drift_rate = curand_normal(&local_state)*V + v;

        /* Set up drift variables. */
        prob_up = .5f*(1+sqrtf(dt)/intra_sv*drift_rate);
        step_size = sqrtf(dt)*intra_sv;
        time = start_delay;
        position = start_point;

        /* Simulate particle movement until threshold is crossed. */
        while (position > 0 & position < a) {
            rand = curand_uniform(&local_state);
            position += ((rand < prob_up)*2 - 1) * step_size;
            time += dt;
        }

        /* Save back state. */
        global_state[idx] = local_state;

        /* Figure out boundary, save result. */
        if (position <= 0) {
            out[idx] = -time;
        }
        else {
            out[idx] = time;
        }
    }

    __global__ void sim_drift_var_thresh(curandState *global_state, float const v, float const V, float const *a, float const z, float const Z, float const t, float const T, float const dt, float const intra_sv, int const a_len, float *out)
    {
        float start_delay, start_point, drift_rate, rand, prob_up, position, step_size, time;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int x_pos = 0;

        curandState local_state = global_state[idx];

        start_delay = curand_uniform(&local_state)*T + (t-T/2);
        start_point = curand_uniform(&local_state)*Z + (z-Z/2);
        drift_rate = curand_normal(&local_state)*V + v;

        prob_up = .5f*(1+sqrtf(dt)/intra_sv*drift_rate);
        step_size = sqrtf(dt)*intra_sv;
        time = 0;
        position = start_point;

        while (fabs(position) < a[x_pos] & time < a_len) {
            rand = curand_uniform(&local_state);
            position += ((rand < prob_up)*2 - 1) * step_size;
            time += dt;
            x_pos++;
        }

        time += start_delay;

        global_state[idx] = local_state;

        if (position <= 0) {
            out[idx] = -time;
        }
        else {
            out[idx] = time;
        }
    }

    }
    """

mod = SourceModule(code, keep=False, no_extern_c=True)

_size = 512

_sim_drift_cuda = mod.get_function("sim_drift")
_sim_drift_var_thresh_cuda = mod.get_function("sim_drift_var_thresh")

_generator = None
_thresh = []
_thresh_gpu = None
_out = None

def sim_drift(v, V, a, z, Z, t, T, size=512, dt=1e-4, update=False, return_gpu=False):
    global _generator, _out, _size
    if _generator is None or update:
        _generator = pycuda.curandom.XORWOWRandomNumberGenerator()
    max_size = _generator.generators_per_block
    if size // max_size > 1:
        print "too big"
    if _out is None or update:
        _out = gpuarray.empty(size, dtype=np.float32)

    _sim_drift_cuda(_generator.state, np.float32(v), np.float32(V), np.float32(a), np.float32(z), np.float32(Z), np.float32(t), np.float32(T), np.float32(dt), np.float32(1), _out, block=(64, 1, 1), grid=(size // 64 + 1, 1))

    if return_gpu:
        return _out
    else:
        return _out.get()


def sim_drift_var_thresh(v, V, a, z, Z, t, T, max_time, size=512, dt=1e-4, update=False, return_gpu=False):
    global _generator, _thresh, _thresh_gpu, _out

    # Init
    if _generator is None or update:
        _generator = pycuda.curandom.XORWOWRandomNumberGenerator()
    max_size = _generator.generators_per_block
    if size // max_size > 1:
        print "too big"
    if _thresh_gpu is None or update or np.any(_thresh != a):
        if isinstance(a, pycuda.gpuarray.GPUArray):
            _thresh_gpu = a
        else:
            _thresh_gpu = gpuarray.to_gpu(a)
    if _out is None or update:
        _out = gpuarray.empty(size, dtype=np.float32)

    _sim_drift_var_thresh_cuda(_generator.state, np.float32(v), np.float32(V), _thresh_gpu, np.float32(z), np.float32(Z), np.float32(t), np.float32(T), np.float32(dt), np.float32(1), np.float32(max_time), _out, block=(64, 1, 1), grid=(size // 64 + 1,1))

    if return_gpu:
        return _out
    else:
        return _out.get()


def gen_weibull_gpu(a, k, l, max_time=5, dt=1e-4):
    max_time = np.float32(max_time)
    x = pycuda.gpuarray.arange(0., max_time, dt, dtype=np.float32)

    # Weibull pdf
    thresh_func_gpu = k / l * (x / l)**(k - 1) * pycuda_exp(-(x / l)**k)
    thresh_func_gpu *= a
    return thresh_func_gpu

thresh_func = gen_weibull_gpu(3, 1, 1)
max_time = 5.

thresh_const = np.ones(max_time/1e-4, dtype=np.float32)
thresh_func_const = pycuda.gpuarray.to_gpu(thresh_const)
print thresh_func_const.get()
plt.plot(thresh_func.get())
plt.plot(-thresh_func.get())

#thresh_func = np.array(a*np.exp(-rate*np.linspace(0, max_time, steps)), dtype=np.float32)
size = 256
out = sim_drift_var_thresh(.5, .1, thresh_func, 0., .1, .3, .1, max_time, size=size, update=True)

plt.figure()
plt.hist(out, bins=40)

out = sim_drift(1, .1, 2, .5, .1, .3, .1, size)
plt.figure()
plt.hist(out, bins=40)

plt.show()

sum_stats_len = 2 * 10 + 1

def compute_stats_vec(data):
    stats_vec = np.empty(sum_stats_len)
    lower = np.abs(data[data<0])
    upper = data[data>0]
    for i, data_resp in enumerate([lower, upper]):
        # mean
        stats_vec[i * 10 + 0] = np.mean(data_resp)
        # std
        stats_vec[i * 10 + 1] = np.std(data_resp)
        # skew
        stats_vec[i * 10 + 2] = sp.stats.skew(data_resp)
        # 7 quantiles
        stats_vec[i * 10 + 3:i * 10 + 9] = stats_vec.extend(sp.stats.mstats.mquantiles(data_resp, np.linspace(0,1,8, endpoint=False)[1:]))

    # Perc of resps
    stats_vec[-1] = len(upper) / len(lower)

    return stats_vec

def synth_likelihood(data, v, V, a_k, a_l, z, Z, t, T, samples=50, drifts_per_sample=512):
    """Compute synthetic likelihood for collapsing threshold wfpt."""
    summary_vecs = np.empty((samples, sum_stats_len))
    thresh = gen_weibull_gpu(a_k, a_l)

    for sample in range(samples):
        rts = sim_drift_var_thresh(v, V, thresh, z, Z, t, T)
        summary_vecs[sample, :] = compute_stats_vec(rts.get())

    summary_data = compute_stats_vec(data)
    mean = np.mean(summary_vecs, axis=0)
    cov = np.cov(summary_vecs)

    # Evaluate synth likelihood
    logp = mv_normal_like(summary_data, mean, cov)

    return logp

WienerVarThresh = pm.stochastic_from_dist(name="Wiener synthetic likelihood",
                                          logp=synth_likelihood,
                                          dtype=np.float32,
                                          mv=False)
