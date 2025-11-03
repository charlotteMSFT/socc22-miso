#!/usr/bin/env python3
'''
This script simulates a MISO system with 1000 jobs on multiple GPUs with MIG partitioning based on the SoCCC 2022 paper: MISO.
'''

import numpy as np
import random
import os
import sys
import time as time_module
import json
import argparse

from pathlib import Path
from copy import deepcopy
from collections import defaultdict

# Get the absolute path of the current script's directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Set base directory relative to script location
base_dir = SCRIPT_DIR / 'socc22-miso'

# Add the simulator directory to path
simulator_path = base_dir / 'mps' / 'scheduler' / 'simulator'
if simulator_path.exists():
    sys.path.insert(0, str(simulator_path))
else:
    print(f"Warning: Simulator path does not exist: {simulator_path}")

# Import utils
try:
    from utils import GPU_Status, get_speedup
except ImportError as e:
    print(f"Error importing utils: {e}")
    print(f"Make sure the path exists: {simulator_path}")
    sys.exit(1)

''' MISO Simulator for large-scale jobs'''
class MISOSimulator:
    """Event-driven simulator for MISO scheduling policy."""
    def __init__(self, num_jobs=1000, num_gpus=8, arrival_rate=60, seed=0, error_mean=0.016, error_std=0.0032, policy='miso'):
        '''
        Initialize the MISO simulator with given parameters.
        
        Arguments:
            num_jobs (int): Number of jobs to simulate.
            num_gpus (int): Number of GPUs available.
            arrival_rate (float): Average time between job arrivals.
            seed (int): Random seed for reproducibility.
            error_mean (float): Mean of the error distribution for job runtimes.
            error_std (float): Standard deviation of the error distribution for job runtimes.
            policy (str): Scheduling policy to use ('miso', 'oracle', 'full', 'static').
        '''
        self.num_jobs = num_jobs
        self.num_gpus = num_gpus
        self.arrival_rate = arrival_rate
        self.seed = seed
        self.error_mean = error_mean
        self.error_std = error_std
        self.policy = policy
        self.migration_overhead = 2.0  # Migration overhead in seconds (ADDED)

        # Set random seed.
        random.seed(seed)
        np.random.seed(seed)

        # Load job models and traces.
        base_path = base_dir / 'mps' / 'scheduler'
        with open(base_path / 'simulator' / 'job_models.json', 'r') as f:
            job_models = json.load(f)

        with open(base_path / 'trace' / 'trace_100.json', 'r') as f:
            job_dict = json.load(f)

        # Generate job workload (repeat 100-job pattern 10 times).
        self.job_runtime = {}
        self.job_model = {}
        for i in range(num_jobs):
            index = i % 100
            self.job_runtime[i] = float(job_dict[str(index)])
            self.job_model[i] = job_models[str(index)]

        # Generate performance data (actual and predicted performance).
        print(f"Loading performance data for seed {seed} with error mean: {error_mean}, error std: {error_std} ...")
        self.perf_actual, self.perf_pred = get_speedup(job_models, error_mean, error_std)

        # Generate job arrivals based on Poisson process.
        self.generate_arrivals()

        # Initialize GPU states.
        self.gpu_states = [GPU_Status(i) for i in range(num_gpus)]

        # Tracking dictionaries.
        self.job_gpu = {}
        self.job_start_time = {}
        self.job_completion_time = {}
        self.job_slice = {}
        self.num_migrations = 0


    def generate_arrivals(self):
        ''' Generate job arrival times based on a Poisson process. '''
        arrival_order = list(range(self.num_jobs))
        random.shuffle(arrival_order)

        self.job_arrivals = {}
        arrival_time = 0
        for job in arrival_order:
            arrival_time += np.random.poisson(self.arrival_rate)
            self.job_arrivals[job] = arrival_time
        
        # Sort by arrival time.
        self.arrival_sequence = sorted(self.job_arrivals.items(), key=lambda x: x[1])
        # Create arrival_queue as an alias for compatibility (ADDED)
        self.arrival_queue = self.arrival_sequence

    """
    GPU Scheduling Policies for MISO Simulator
    Includes: MISO, Oracle, Full GPU, and Static Partitioning policies
    """

    def find_best_gpu_miso(self, job, current_time):
        '''
        MISO Policy: find the best GPU partition to minimize mean degradation.
        Uses predicted performance data (which scheduler sees).
        This represents a realistic scheduler with prediction errors.
        '''
        best_gpu = None
        best_config = None
        min_degradation = float('inf')

        for gpu in self.gpu_states:
            # Check if GPU can accept more jobs (max 7 MIG slices on a A100 GPU).
            if len(gpu.partition) >= 7:
                continue

            try:
                # Try to optimize with this new job on this GPU.
                test_gpu = deepcopy(gpu)
                partition, assignment, code = test_gpu.miso_optimize(job, self.perf_pred)

                # Calculate mean degradation using predicted performance.
                degradations = []
                for j, slice_type in zip(assignment, [test_gpu.num_to_str[p] for p in partition]):
                    job_mapped = j % 100
                    if slice_type in self.perf_pred[job_mapped]:
                        degradation = self.perf_pred[job_mapped][slice_type] / self.perf_pred[job_mapped]['7g.40gb']
                        degradations.append(degradation)

                # Only update best GPU if we found valid degradations
                if degradations:
                    mean_degradation = np.mean(degradations)
                    if mean_degradation < min_degradation:
                        min_degradation = mean_degradation
                        best_gpu = gpu
                        best_config = (partition, assignment, code)
            
            except Exception as e:
                print(f"Error optimizing GPU {gpu.gpu_id} for job {job}: {e}")
                continue

        return best_gpu, best_config


    def find_best_gpu_oracle(self, job, current_time):
        '''
        Oracle Policy: find the best GPU partition to minimize mean degradation.
        Uses ACTUAL performance data (no prediction error).
        This represents the theoretical best performance - an upper bound for comparison.
        The oracle has perfect knowledge of actual job performance.
        '''
        best_gpu = None
        best_config = None
        min_degradation = float('inf')

        for gpu in self.gpu_states:
            # Check if GPU can accept more jobs (max 7 MIG slices on a A100 GPU).
            if len(gpu.partition) >= 7:
                continue

            try:
                # Try to optimize with this new job on this GPU.
                # Key difference from MISO: use self.perf_actual instead of self.perf_pred
                test_gpu = deepcopy(gpu)
                partition, assignment, code = test_gpu.miso_optimize(job, self.perf_actual)

                # Calculate mean degradation using ACTUAL performance data.
                degradations = []
                for j, slice_type in zip(assignment, [test_gpu.num_to_str[p] for p in partition]):
                    job_mapped = j % 100
                    if slice_type in self.perf_actual[job_mapped]:
                        degradation = self.perf_actual[job_mapped][slice_type] / self.perf_actual[job_mapped]['7g.40gb']
                        degradations.append(degradation)

                # Only update best GPU if we found valid degradations
                if degradations:
                    mean_degradation = np.mean(degradations)
                    if mean_degradation < min_degradation:
                        min_degradation = mean_degradation
                        best_gpu = gpu
                        best_config = (partition, assignment, code)
            
            except Exception as e:
                print(f"Error optimizing GPU {gpu.gpu_id} for job {job}: {e}")
                continue

        return best_gpu, best_config


    def find_best_gpu_full(self, job, current_time):
        '''
        Full GPU Policy: no partitioning, one job per GPU.
        This is the traditional approach where each job gets exclusive access to a full GPU.
        Simple baseline policy for comparison.
        '''
        for gpu in self.gpu_states:
            if gpu.jobs == ['idle']:
                return gpu, None
        return None, None


    def find_best_gpu_static(self, job, current_time):
        '''
        Static Partitioning Policy: uses fixed, predetermined GPU partitions.
        Finds the largest available partition slice that can accommodate the job.
        Does not dynamically reconfigure partitions based on workload.
        '''
        best_gpu = None
        max_slice = 0
        
        for gpu in self.gpu_states:
            if 'idle' in gpu.jobs:
                max_size, max_ind = gpu.max_inactive_slice
                if max_size and max_size > max_slice:
                    slice_type = gpu.num_to_str[max_size]
                    job_mapped = job % 100
                    if slice_type in self.perf_pred[job_mapped]:
                        max_slice = max_size
                        best_gpu = gpu
        
        return best_gpu, None


    def schedule_job(self, job, current_time):
        '''
        Main scheduling function that dispatches to the appropriate policy.
        
        Arguments:
            job (int): Job ID to schedule.
            current_time (float): Current simulation time.
        
        Returns:
            tuple: (success, completion_time) where success is True if job was scheduled,
                   and completion_time is when the job will complete.
        '''
        # Find best GPU based on policy
        if self.policy == 'miso':
            best_gpu, config = self.find_best_gpu_miso(job, current_time)
        elif self.policy == 'oracle':
            best_gpu, config = self.find_best_gpu_oracle(job, current_time)
        elif self.policy == 'full':
            best_gpu, config = self.find_best_gpu_full(job, current_time)
        elif self.policy == 'static':
            best_gpu, config = self.find_best_gpu_static(job, current_time)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        if best_gpu is None:
            return False, None
        
        # Apply the best configuration to the selected GPU.
        if self.policy in ["miso", "oracle"]:
            if config:
                partition, assignment, code = config
                best_gpu.implement_miso_opt(partition, assignment, code, 
                                           self.perf_actual if self.policy == 'oracle' else self.perf_pred)
        
        elif self.policy == "full":
            best_gpu.jobs = [job]
            best_gpu.max_allowed = 'full'

        elif self.policy == "static":
            max_size, max_ind = best_gpu.max_inactive_slice
            best_gpu.jobs[max_ind] = job
            best_gpu.static_max_slice()

        # Record assignment
        self.job_gpu[job] = best_gpu.gpu_id  # Changed from best_gpu.index to best_gpu.gpu_id
        self.job_start_time[job] = current_time
        self.job_slice[job] = best_gpu.get_job_slices(job)

        # Calculate completion time using actual performance data.
        job_mapped = job % 100
        slice_type = self.job_slice[job]
        base_time = self.job_runtime[job]

        # Actual performance.
        if slice_type in self.perf_actual[job_mapped]:
            speedup = self.perf_actual[job_mapped][slice_type]
            completion_time = current_time + (base_time / speedup)
        else:
            # Fallback to full GPU performance if slice type not found.
            speedup = self.perf_actual[job_mapped]['7g.40gb']
            completion_time = current_time + base_time

        self.job_completion_time[job] = completion_time
        return True, completion_time
    
    def handle_job_completion(self, job, current_time):
        """
        Remove completed job and trigger idle optimization
        """
        gpu_index = self.job_gpu[job]
        gpu = self.gpu_states[gpu_index]
        
        # Find and remove job
        try:
            job_index = gpu.jobs.index(job)
            gpu.jobs[job_index] = 'idle'
        except ValueError:
            pass  # Job already removed
        
        # Trigger idle partition optimization for MISO/Oracle
        if self.policy in ['miso', 'oracle'] and 'idle' in gpu.jobs and len(gpu.active_jobs) > 0:
            perf_dict = self.perf_actual if self.policy == 'oracle' else self.perf_pred
            migrations, migrated_jobs = gpu.idle_partition_optimize(perf_dict)
            self.num_migrations += migrations
            
            # Update completion times for migrated jobs
            for mig_job in migrated_jobs:
                if mig_job in self.job_completion_time:
                    old_completion = self.job_completion_time[mig_job]
                    time_remaining = old_completion - current_time
                    
                    # Recalculate with new slice
                    new_slice = gpu.get_job_slices(mig_job)  # Changed to plural
                    job_mapped = mig_job % 100
                    
                    if new_slice in self.perf_actual[job_mapped]:
                        new_speedup = self.perf_actual[job_mapped][new_slice]
                        old_slice = self.job_slice[mig_job]
                        old_speedup = self.perf_actual[job_mapped][old_slice]
                        
                        # Adjust remaining time
                        time_remaining_adjusted = time_remaining * (old_speedup / new_speedup)
                        new_completion = current_time + time_remaining_adjusted + self.migration_overhead
                        
                        self.job_completion_time[mig_job] = new_completion
                        self.job_slice[mig_job] = new_slice

    def run(self):
        """
        Main event-driven simulation loop
        """
        print(f"\nRunning {self.policy.upper()} policy simulation...")
        print(f"Jobs: {self.num_jobs}, GPUs: {self.num_gpus}, Seed: {self.seed}")
        
        current_time = 0
        queue_index = 0
        pending_jobs = []  # Jobs that couldn't be scheduled immediately
        running_jobs = {}  # {job: completion_time}
        completed_jobs = []
        
        start_wall_time = time_module.time()
        
        while len(completed_jobs) < self.num_jobs:
            # Process arrivals
            while queue_index < len(self.arrival_queue):
                job, arrival_time = self.arrival_queue[queue_index]
                if arrival_time <= current_time:
                    pending_jobs.append(job)
                    queue_index += 1
                else:
                    break
            
            # Try to schedule pending jobs
            newly_scheduled = []
            for job in pending_jobs:
                success, completion = self.schedule_job(job, current_time)
                if success:
                    running_jobs[job] = completion
                    newly_scheduled.append(job)
            
            # Remove scheduled jobs from pending
            for job in newly_scheduled:
                pending_jobs.remove(job)
            
            # Check for completions
            completed_now = []
            for job, comp_time in list(running_jobs.items()):
                if comp_time <= current_time:
                    completed_jobs.append(job)
                    completed_now.append(job)
                    self.handle_job_completion(job, current_time)
            
            # Remove completed jobs
            for job in completed_now:
                del running_jobs[job]
            
            # Progress indicator
            if len(completed_jobs) % 100 == 0 and len(completed_jobs) > 0:
                elapsed = time_module.time() - start_wall_time
                print(f"  Completed: {len(completed_jobs)}/{self.num_jobs} jobs ({elapsed:.1f}s)")
            
            # Advance time to next event
            next_events = []
            if running_jobs:
                next_events.append(min(running_jobs.values()))
            if queue_index < len(self.arrival_queue):
                next_events.append(self.arrival_queue[queue_index][1])
            
            if next_events:
                current_time = min(next_events)
            else:
                if pending_jobs:
                    # Deadlock - shouldn't happen in normal operation
                    print(f"WARNING: {len(pending_jobs)} jobs pending but no events scheduled")
                    current_time += 10
                else:
                    break
        
        wall_time = time_module.time() - start_wall_time
        print(f"  Simulation completed in {wall_time:.1f}s")
        
        return self.calculate_metrics(completed_jobs)
    
    def calculate_metrics(self, completed_jobs):
        """Calculate performance metrics"""
        jcts = []  # Job Completion Times
        queueing_delays = []
        
        for job in completed_jobs:
            arrival = self.job_arrivals[job]
            start = self.job_start_time[job]
            completion = self.job_completion_time[job]
            
            jct = completion - arrival
            queueing_delay = start - arrival
            
            jcts.append(jct)
            queueing_delays.append(queueing_delay)
        
        makespan = max(self.job_completion_time.values()) if self.job_completion_time else 0
        
        metrics = {
            'policy': self.policy,
            'seed': self.seed,
            'num_jobs': self.num_jobs,
            'num_gpus': self.num_gpus,
            'mean_jct': float(np.mean(jcts)),
            'median_jct': float(np.median(jcts)),
            'p95_jct': float(np.percentile(jcts, 95)),
            'p99_jct': float(np.percentile(jcts, 99)),
            'std_jct': float(np.std(jcts)),
            'mean_queueing_delay': float(np.mean(queueing_delays)),
            'makespan': float(makespan),
            'throughput': self.num_jobs / makespan if makespan > 0 else 0,
            'num_migrations': self.num_migrations,
        }
        
        return metrics


def run_single_experiment(args):
    """Run a single experiment with given parameters"""
    simulator = MISOSimulator(
        num_jobs=args.num_jobs,
        num_gpus=args.num_gpus,
        arrival_rate=args.arrival,
        seed=args.seed,
        error_mean=args.error_mean,
        error_std=args.error_std,
        policy=args.policy
    )
    
    metrics = simulator.run()
    
    return metrics


def run_multiple_trials(args):
    """Run multiple trials with different seeds"""
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"Running {args.num_trials} trials of {args.policy.upper()} policy")
    print(f"{'='*60}")
    
    for trial in range(args.num_trials):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.seed = args.seed + trial
        
        print(f"\n--- Trial {trial+1}/{args.num_trials} (seed={trial_args.seed}) ---")
        
        metrics = run_single_experiment(trial_args)
        all_results.append(metrics)
        
        # Print summary
        print(f"  Mean JCT: {metrics['mean_jct']:.1f}s")
        print(f"  Makespan: {metrics['makespan']:.1f}s")
        print(f"  Migrations: {metrics['num_migrations']}")
        
        # Save intermediate results
        if (trial + 1) % 10 == 0 or trial == args.num_trials - 1:
            output_file = f"results_{args.policy}_trial_{trial+1}.json"
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved intermediate results to {output_file}")
    
    # Calculate aggregate statistics
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.num_trials} trials)")
    print(f"{'='*60}")
    
    mean_jcts = [r['mean_jct'] for r in all_results]
    makespans = [r['makespan'] for r in all_results]
    migrations = [r['num_migrations'] for r in all_results]
    
    aggregate = {
        'policy': args.policy,
        'num_trials': args.num_trials,
        'num_jobs_per_trial': args.num_jobs,
        'num_gpus': args.num_gpus,
        'avg_mean_jct': float(np.mean(mean_jcts)),
        'std_mean_jct': float(np.std(mean_jcts)),
        'avg_makespan': float(np.mean(makespans)),
        'std_makespan': float(np.std(makespans)),
        'avg_migrations': float(np.mean(migrations)),
        'all_trials': all_results
    }
    
    print(f"Average Mean JCT: {aggregate['avg_mean_jct']:.1f} ± {aggregate['std_mean_jct']:.1f}s")
    print(f"Average Makespan: {aggregate['avg_makespan']:.1f} ± {aggregate['std_makespan']:.1f}s")
    print(f"Average Migrations: {aggregate['avg_migrations']:.1f}")
    
    # Save final results
    output_file = f"results_{args.policy}_final.json"
    with open(output_file, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nFinal results saved to {output_file}")
    
    return aggregate


def main():
    parser = argparse.ArgumentParser(description='MISO Large-Scale Simulator')
    parser.add_argument('--num_jobs', type=int, default=1000, help='Number of jobs to simulate')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--arrival', type=int, default=60, help='Mean inter-arrival time (seconds)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--error_mean', type=float, default=0.016, help='Prediction error mean')
    parser.add_argument('--error_std', type=float, default=0.0032, help='Prediction error std dev')
    parser.add_argument('--policy', type=str, default='miso', 
                       choices=['miso', 'static', 'full', 'oracle'],
                       help='Scheduling policy')
    parser.add_argument('--num_trials', type=int, default=1, 
                       help='Number of trials to run (each with different seed)')
    
    args = parser.parse_args()
    
    if args.num_trials > 1:
        run_multiple_trials(args)
    else:
        metrics = run_single_experiment(args)
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:.2f}")
            else:
                print(f"{key:25s}: {value}")
        
        # Save results
        output_file = f"results_{args.policy}_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
