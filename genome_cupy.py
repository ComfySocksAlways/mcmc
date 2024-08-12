import hopsy
from PolyRound.api import PolyRoundApi
import os
import time
import numpy as np
import copy
import argparse
import cupy as cp

def main(n_samples, verbose):
    model_path = os.path.join("hopsy","examples","test_data", "e_coli_core.xml")
    polytope = PolyRoundApi.sbml_to_polytope(model_path)
    problem = hopsy.Problem(polytope.A, polytope.b)
    problem = hopsy.add_box_constraints(problem, upper_bound=10_000, lower_bound=-10_000, simplify=True)
    start = time.perf_counter()
    problem = hopsy.round(problem)
    print("Computing rounding transformation took", time.perf_counter()-start,"seconds")

    seed = 511
    chains, rngs = hopsy.setup(problem, seed, n_chains=4)
    # Either use thinning rule, see  10.1371/journal.pcbi.1011378
    # or use one-shot transformation (for expert users). We show one-shot transformation at the end.
    thinning = int(1./6*problem.transformation.shape[1])

    # Sample without Transform
    assert problem.transformation is not None
    # deep copy enures that we do not edit the original problem
    problem2 = copy.deepcopy(problem)
    problem2.transformation=None
    problem2.shift=None
    seed = 512
    chains, rngs = hopsy.setup(problem2, seed, n_chains=4)
    # thinning is still advised when hard drive memory is limisted to not to store too many samples 
    thinning = int(1./6*problem.A.shape[1])  

    start = time.perf_counter()
    accrate, sample_stack = hopsy.sample(chains, rngs, n_samples, thinning=thinning, n_procs=4)
    # accrate is 1 for uniform samples with the default chains given by hopsy.setup()
    print("sampling took", time.perf_counter()-start,"seconds")

    
    shift_t_pl = np.array([problem.shift])
    start_trafo = time.perf_counter()
    full_samples_pl = np.zeros((len(chains), n_samples, sample_stack.shape[2]))

    # Device arrays
    d_transformation = cp.asarray(problem.transformation.T)
    d_sample_stack = cp.asarray(sample_stack)
    d_shift = cp.asarray(shift_t_pl)
    d_full_samples = cp.asarray(full_samples_pl)
    for i in range(len(chains)):
        d_full_samples[i] = d_sample_stack[i]@d_transformation + d_shift
    full_samples_pl = cp.asnumpy(d_full_samples)
    print("transformation took", time.perf_counter()-start_trafo,"seconds")

    if verbose:
        rhat = np.max(hopsy.rhat(full_samples_pl))
        print("rhat:", rhat)
        ess = np.min(hopsy.ess(full_samples_pl)) / len(chains)
        print("ess:", ess)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sampling and transformation with specified parameters.")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    
    args = parser.parse_args()
    main(args.n_samples, args.verbose)