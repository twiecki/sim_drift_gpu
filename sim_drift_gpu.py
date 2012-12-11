from __future__ import division

import timeit
import pycuda.compiler
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.curandom
from pycuda import cumath

from pycuda.cumath import exp as pycuda_exp
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from kabuki.distributions import scipy_stochastic
import time

import numpy as np
import scipy as sp
import scipy.stats
import pymc as pm

import hddm

from thrust_sort import sort as sort_gpu

code = """
    #include <curand_kernel.h>

    extern "C"
    {

    __global__ void fill_delta(float *out, const int n)
    {
        const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        const int delta = blockDim.x*gridDim.x;

        //out[tidx] = tidx;

        for (int idx = tidx; idx < n; idx += delta)
        {
            out[idx] = tidx;
        }
    }

    __global__ void fill_lb_quantiles(const float *data, const float *quantiles, const float *lb, const float *ub, float *out)
    {
        const int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int q_idx;
        q_idx = static_cast<int> (lb[0] - floor(quantiles[idx] * lb[0]));
        out[idx] = data[q_idx];
    }

    __global__ void fill_ub_quantiles(const float *data, const float *quantiles, const float *lb, const float *ub, float *out)
    {
        const int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int q_idx;
        q_idx = static_cast<int> (floor(quantiles[idx] * ub[0]) + lb[0]);
        out[idx] = data[q_idx];
    }

    __global__ void sim_drift(curandState *global_state, float const v, float const V, float const a, float const z, float const Z, float const t, float const T, float const dt, float const intra_sv, float *out, const int n)
    {
        float start_delay, start_point, drift_rate, rand, prob_up, position, step_size, time;

        float sqrt_dt = sqrtf(dt);

        const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        const int delta = blockDim.x*gridDim.x;

        curandState local_state = global_state[tidx];

        for (int idx = tidx; idx < n; idx += delta)
        {
            /* Sample variability parameters. */
            if (T != 0)
                start_delay = curand_uniform(&local_state)*T + (t-T/2);
            else
                start_delay = t;

            if (Z != 0)
                start_point = (curand_uniform(&local_state)*Z + (z-Z/2))*a;
            else
                start_point = z*a;

            if (V != 0)
                 drift_rate = curand_normal(&local_state)*V + v;
            else
                 drift_rate = v;

            /* Set up drift variables. */
            prob_up = .5f*(1+sqrtf(dt)/intra_sv*drift_rate);
            step_size = sqrtf(dt)*intra_sv;
            time = start_delay;
            position = start_point;

            /* Simulate brownian motion until threshold is crossed. */
            while (position > 0 & position < a) {
                rand = curand_uniform(&local_state);
                position += ((rand < prob_up)*2 - 1) * step_size;
                time += dt;
            }

            /* Figure out boundary, save result. */
            if (position <= 0) {
                out[idx] = -time;
            }
            else {
                out[idx] = time;
            }

        }
        /* Save back state. */
        global_state[tidx] = local_state;
    }

    __global__ void sim_drift_var_thresh(curandState *global_state, float const v, float const V, float const *a, float const z, float const Z, float const t, float const T, float const dt, float const intra_sv, int const a_len, float *out, const int n)
    {
        float start_delay, start_point, drift_rate, rand, prob_up, position, step_size, time;
        int x_pos;

        const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
        const int delta = blockDim.x*gridDim.x;

        for (int idx = tidx; idx < n; idx += delta)
        {

            x_pos = 0;

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

            if (position <= 0)
                out[idx] = -time;
            else
                out[idx] = time;

            global_state[idx] = local_state;

        }
    }

/*    __global__ void sim_drift_switch(curandState *global_state, float const vpp, float const vcc, float const a, float const z, float const t, float const tcc, float const dt, float const intra_sv, float *out)
    {
        float start_delay, start_point, rand, prob_up_pp, prob_up_cc, position, step_size, time;
        int idx = blockIdx.x*blockDim.x + threadIdx.x;

        curandState local_state = global_state[idx];

        start_delay = t;
        start_point = z;

        prob_up_pp = .5f*(1+sqrtf(dt)/intra_sv*vpp);
        prob_up_cc = .5f*(1+sqrtf(dt)/intra_sv*vcc);

        step_size = sqrtf(dt)*intra_sv;
        time = 0;
        position = start_point;

        while (fabs(position) < a) {
            rand = curand_uniform(&local_state);
            if time < tcc {
                position += ((rand < prob_up_pp)*2 - 1) * step_size;
            }
            else {
                position += ((rand < prob_up_cc)*2 - 1) * step_size;
            time += dt;
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
*/
    }
    """

mod = SourceModule(code, keep=False, no_extern_c=True)

_sim_drift_cuda = mod.get_function("sim_drift")
_sim_drift_var_thresh_cuda = mod.get_function("sim_drift_var_thresh")
fill_lb_quantiles = mod.get_function("fill_lb_quantiles")
fill_ub_quantiles = mod.get_function("fill_ub_quantiles")
fill_delta = mod.get_function("fill_delta")

_generator = None
_thresh = []
_thresh_gpu = None
_out = None
_generators_per_block = 0
_block_count = 0

def test_fill_delta():
    size = 512
    out = gpuarray.zeros(size, np.float32)
    fill_delta(out, np.uint32(size), block=(512, 1, 1), grid=(2, 1))

    print out.get()


def sim_drift(v, V, a, z, Z, t, T, size=512, dt=1e-4, update=False, return_gpu=False):
    global _generator, _out, _generators_per_block, _block_count
    size = np.long(size)
    if _generator is None or update:
        _generator = pycuda.curandom.XORWOWRandomNumberGenerator()
        _block_count = _generator.block_count
        _generators_per_block = _generator.generators_per_block
    if _out is None or update:
        _out = gpuarray.empty(size, dtype=np.float32)

    _sim_drift_cuda(_generator.state, np.float32(v), np.float32(V), np.float32(a), np.float32(z), np.float32(Z), np.float32(t), np.float32(T), np.float32(dt), np.float32(1), _out, np.uint32(size), block=(_generators_per_block, 1, 1), grid=(_block_count, 1))

    if return_gpu:
        return _out
    else:
        return _out.get()


def sim_drift_var_thresh(v, V, a, z, Z, t, T, max_time, size=512, dt=1e-4, update=False, return_gpu=False):
    global _generator, _thresh, _thresh_gpu, _out

    # Init
    if _generator is None or update:
        _generator = pycuda.curandom.XORWOWRandomNumberGenerator()
    if _thresh_gpu is None or update or np.any(_thresh != a):
        if isinstance(a, pycuda.gpuarray.GPUArray):
            _thresh_gpu = a
        else:
            _thresh_gpu = gpuarray.to_gpu(a)
    if _out is None or update:
        _out = gpuarray.empty(size, dtype=np.float32)

    _sim_drift_var_thresh_cuda(_generator.state, np.float32(v), np.float32(V), _thresh_gpu, np.float32(z), np.float32(Z), np.float32(t), np.float32(T), np.float32(dt), np.float32(1), np.float32(max_time), _out, np.uint32(size), block=(_generator.block_count, 1, 1), grid=(size // 64 + 1,1))

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

class wfpt_gpu_gen(hddm.likelihoods.wfpt_gen):
    sampling_method = 'gpu'

    def _rvs(self, v, V, a, z, Z, t, T):
        param_dict = {'v':v, 'z':z, 't':t, 'a':a, 'Z':Z, 'V':V, 'T':T}
        if self.sampling_method == 'gpu':
            sampled_rts = sim_drift(v, V, a, z, Z, t, T, size=self._size, dt=self.dt, return_gpu=True)
        else:
            sampled_rts = hddm.generate.gen_rts(param_dict, method=self.sampling_method, samples=self._size, dt=self.dt)
        return sampled_rts[:self._size]

wfpt_gpu_like = scipy_stochastic(wfpt_gpu_gen, name='wfpt_gpu')


def main():
    thresh_func = gen_weibull_gpu(3, 1, 1)
    max_time = 5.
    dt = 1e-4

    thresh_const = np.ones(max_time/dt, dtype=np.float32)
    thresh_func_const = pycuda.gpuarray.to_gpu(thresh_const)
    print thresh_func_const.get()
    plt.plot(np.arange(0, max_time, dt), thresh_func.get())
    plt.plot(np.arange(0, max_time, dt), -thresh_func.get())

    #thresh_func = np.array(a*np.exp(-rate*np.linspace(0, max_time, steps)), dtype=np.float32)
    size = 512
    out = sim_drift_var_thresh(.5, .1, thresh_func, 0., .1, .3, .1, max_time, size=size, update=True)

    plt.figure()
    plt.hist(out, bins=40)

    out = sim_drift(1, .1, 2, .5, .1, .3, .1, size)
    plt.figure()
    plt.hist(out, bins=40)

    plt.show()


quantiles = gpuarray.to_gpu(np.array((0, .1, .3, .5, .7, .9, 1), dtype=np.float32))
q_lb = gpuarray.empty_like(quantiles)
q_ub = gpuarray.empty_like(quantiles)
probs = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
mask = gpuarray.zeros(2048, dtype=np.float32)

def create_quantiles(data, params):
    global quantiles, q_lb, q_ub, mask
    sort_gpu(data)

    if mask.shape != data.shape:
        mask = gpuarray.zeros_like(data)

    n_lb = gpuarray.sum(data < mask)
    n_ub = gpuarray.sum(data > mask)

    fill_lb_quantiles(data, quantiles, n_lb, n_ub, q_lb, block=(quantiles.shape[0], 1, 1))
    fill_ub_quantiles(data, quantiles, n_lb, n_ub, q_ub, block=(quantiles.shape[0], 1, 1))
    q_lb = q_lb.reverse()

    p_ub = n_ub / (n_ub + n_lb)

    del n_lb, n_ub

    return data, q_lb.get(), q_ub.get(), probs*(1-p_ub.get()), probs*p_ub.get()


def gen_summary_stats(data):
    lb = data[data < mask]
    ub = data[data > mask]

    prob_ub = ub / lb
    n_ub = ub.size
    n_lb = lb.size
    mean_ub = gpuarray.sum(ub) / n_ub
    mean_lb = gpuarray.sum(lb) / n_lb
    var_ub = (ub - mean_ub)**2 / n_ub
    var_lb = (lb - mean_lb)**2 / n_lb
    #skew = (data - mean)**3 / var**(3/2.)




def multinomial_like(value, v, V, a, z, Z, t, T):
    # sample wfpts
    data_gpu = sim_drift(v, V, a, z, Z, t, T, size=2048, dt=1e-4,
                         return_gpu=True)
    # calculate quantiles
    data, q_lb, q_ub, p_lb, p_ub = create_quantiles(data_gpu, {'v': v, 'V': V, 'a': a, 'z': Z, 't': t, 'T': T})

    q_lb[0] = -np.inf
    q_lb[-1] = 0
    q_ub[0] = 0
    q_ub[-1] = np.inf
    # calucate histogram on data
    n_lb = np.histogram(value, bins=q_lb)[0]
    n_ub = np.histogram(value, bins=q_ub)[0]

    # calculate log-likelihood
    logp = np.sum(n_lb * np.log(p_lb)) + np.sum(n_ub * np.log(p_ub))
    if logp == 0:
        print locals()

    return logp

WfptSimLikelihood = pm.stochastic_from_dist(name="Wfpt simulated multinomial likelihood based on quantiles",
                                            logp=multinomial_like,
                                            dtype=np.float,
                                            mv=False)

import hddm

class HDDMSim(hddm.HDDM):
    def get_bottom_node(self, param, params):
        """Create and return the wiener likelihood distribution
        supplied in 'param'.

        'params' is a dictionary of all parameters on which the data
        depends on (i.e. condition and subject).

        """
        if param.name == 'wfpt':
            return WfptSimLikelihood(param.full_name,
                                     value=param.data['rt'].flatten(),
                                     v=params['v'],
                                     a=params['a'],
                                     z=self.get_node('z',params),
                                     t=params['t'],
                                     Z=self.get_node('Z',params),
                                     T=self.get_node('T',params),
                                     V=self.get_node('V',params),
                                     observed=True)

        else:
            raise KeyError, "Groupless parameter named %s not found." % param.name

def test_param_recovery():
    params = hddm.generate.gen_rand_params()
    print params
    data, subj_params = hddm.generate.gen_rand_data(params, samples=500)
    print subj_params

    m = HDDMSim(data)
    m.map()
    for name, param in m.params_include.iteritems():
        #print param.group_nodes
        if not param.is_bottom_node:
            print "%s: %f\n" % (name, param.group_nodes[''].value)
    #m.sample(2000, burn=1000)

    #m.print_stats()




def measure_speed_cpu(size):
    sampler.sampling_method = 'drift'
    t0 = time.time()
    data = sampler.rvs(params['v'], params['V'], params['a'], params['z'],
                params['Z'], params['t'], params['T'], size=size)
    np.sort(data)
    cpu_speed = (time.time()-t0)
    print "CPU: %f" % cpu_speed

    return cpu_speed

def measure_speed():
    repeats = 10
    x = np.arange(100, 500, 100)
    speeds = np.empty_like(x)
    for i, size in enumerate(x):
        t_cpu = timeit.timeit('sim_drift_gpu.measure_speed_cpu(%i)' % (size), setup="import sim_drift_gpu", number=20)
        t_gpu = timeit.timeit('sim_drift_gpu.measure_speed_gpu(%i)' % (size), setup="import sim_drift_gpu", number=20)
        speeds[i] = t_cpu/t_gpu

    plt.plot(x, speeds)
    plt.xlabel("sample size")
    plt.ylabel("Speed-up of GPU")
    plt.show()

if __name__ == '__main__':
    size=512
    params = hddm.generate.gen_rand_params()
    sampler.sampling_method = 'gpu'
    t0 = time.time()
    data_gpu = sim_drift(params['v'], params['V'], params['a'],
                         params['z'], params['Z'], params['t'],
                         params['T'], size=size, dt=1e-4,
                         return_gpu=True)

    data, q_lb, q_ub, p_lb, p_ub = create_quantiles(data_gpu, size)
    gpu_speed = (time.time()-t0)
    print gpu_speed

    data = data.get()
    print p_lb, p_ub
    print np.sum(p_lb) + np.sum(p_ub)
    lb_bins = np.histogram(data[data<0], bins=q_lb.get())[0]
    ub_bins = np.histogram(data[data>0], bins=q_ub.get())[0]
    print lb_bins, ub_bins

