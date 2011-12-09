import pycuda.driver as cuda
import pycuda.compiler
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.curandom
from pycuda.compiler import SourceModule

import numpy as np

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

        while (fabs(position) > a[x_pos] & x_pos < a_len) {
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

_size = 128

_sim_drift_cuda = mod.get_function("sim_drift")
_sim_drift_var_thresh_cuda = mod.get_function("sim_drift_var_thresh")

_generator = None
_thresh = []
_thresh_gpu = None
_out = None

def sim_drift(v, V, a, z, Z, t, T, size, dt=1e-4, update=False):
    global _generator, _out, _size
    if size != size:
        update = True
    if _generator is None or update:
        _generator = pycuda.curandom.XORWOWRandomNumberGenerator()
    if _out is None or update:
        _out = gpuarray.empty(size, dtype=np.float32)

    _sim_drift_cuda(_generator.state, np.float32(v), np.float32(V), np.float32(a), np.float32(z), np.float32(Z), np.float32(t), np.float32(T), np.float32(dt), np.float32(1), _out, block=(64,1,1), grid=(size//64+1,1))

    return _out.get()

def sim_drift_var_thresh(v, V, a, z, Z, t, T, max_time, size, dt=1e-4, update=False):
    global _generator, _thresh, _thresh_gpu, _out

    # Init
    if size != _size:
        update = True
    if _generator is None or update:
        _generator = pycuda.curandom.XORWOWRandomNumberGenerator()
    if _thresh_gpu is None or update:
        _thresh_gpu = gpuarray.to_gpu(a)
    if np.any(_thresh != a) or update:
        # Threshold function changed from before
        _thresh_gpu = gpuarray.to_gpu(a)
        _thresh = a
    if _out is None or update:
        _out = gpuarray.empty(size, dtype=np.float32)

    _sim_drift_var_thresh_cuda(_generator.state, np.float32(v), np.float32(V), _thresh_gpu, np.float32(z), np.float32(Z), np.float32(t), np.float32(T), np.float32(dt), np.float32(1), np.float32(max_time), _out, block=(64,1,1), grid=(size//64+1,1))

    return _out.get()

dt = 1e-4
max_time = 5.
steps = max_time/dt
a = 2
rate = .5
thresh_func = np.array(a*np.exp(-rate*np.linspace(0, max_time, steps)), dtype=np.float32)
size = 128
print sim_drift_var_thresh(0, .1, thresh_func, 0., .1, .3, .1, max_time, size)

print sim_drift(1, .1, 2, .5, .1, .3, .1, size)

print sim_drift(1, .1, 2, .5, .1, .3, .1, size)