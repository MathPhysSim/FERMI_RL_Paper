#!/usr/bin/env python3
"""
Recreate the ORIGINAL paper figures exactly from scripts and data.

Each figure function is copied verbatim from the original plotting scripts,
with only path adjustments for standalone execution:
  - NAF: show_progress_presi_gsi.py
  - AE-DYNA: AEdyna_clean_test.py:plot_observables
  - ME-TRPO: new_ME_TRPO_read_data_presi_gsi.py

The only intentional change: convergence plot shows V only (no Bellman error).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(PROJ, "paper", "Figures")
NAF = os.path.join(PROJ, "data", "Data_Experiments", "2020_07_20_NAF@FERMI")
DYNA = os.path.join(PROJ, "data", "Data_Experiments", "2020_11_05_AE_Dyna@FERMI")
TRPO = os.path.join(PROJ, "data", "Data_Experiments", "2020_10_06_ME_TRPO_stable@FERMI")


# ═══════════════════════════════════════════════════════════════════════════════
# NAF — copied from show_progress_presi_gsi.py
# ═══════════════════════════════════════════════════════════════════════════════

def load_pickle_logging(file_name):
    """Exact copy of show_progress_presi_gsi.py:load_pickle_logging, path-adjusted."""
    directory = os.path.join(NAF, file_name, 'data/')
    files = []
    for f in os.listdir(directory):
        if 'trajectory_data' in f and 'pkl' in f:
            files.append(f)
    files.sort()
    with open(directory + files[-1], 'rb') as f:
        states = pickle.load(f)
        actions = pickle.load(f)
        rewards = pickle.load(f)
        dones = pickle.load(f)
    return states, actions, rewards, dones


def load_pickle_final(file_name):
    """Exact copy of show_progress_presi_gsi.py:load_pickle_final, path-adjusted."""
    file = os.path.join(NAF, file_name, 'plot_data_0.pkl')
    with open(file, 'rb') as f:
        rews = pickle.load(f)
        inits = pickle.load(f)
        losses = pickle.load(f)
        v_s = pickle.load(f)
    return rews, inits, losses, v_s


def read_rewards(rewards0):
    """Exact copy of show_progress_presi_gsi.py:read_rewards."""
    iterations_all = []
    final_rews_all = []
    mean_rews_all = []
    for k in range(len(rewards0)):
        rewards = rewards0[k]
        iterations = []
        final_rews = []
        mean_rews = []
        for i in range(len(rewards)):
            if len(rewards[i]) > 0:
                final_rews.append(rewards[i][len(rewards[i]) - 1])
                iterations.append(len(rewards[i]))
                try:
                    mean_rews.append(np.sum(rewards[i][1:]))
                except:
                    mean_rews.append([])
        iterations_all.append(iterations)
        final_rews_all.append(final_rews)
        mean_rews_all.append(mean_rews)
    # Truncate to minimum length across runs
    min_len = min(len(a) for a in iterations_all)
    iterations_all = [a[:min_len] for a in iterations_all]
    final_rews_all = [a[:min_len] for a in final_rews_all]
    mean_rews_all = [a[:min_len] for a in mean_rews_all]
    iterations = np.mean(np.array(iterations_all), axis=0)
    final_rews = np.mean(np.array(final_rews_all), axis=0)
    mean_rews = np.mean(np.array(mean_rews_all), axis=0)
    return iterations, final_rews, mean_rews


def plot_results(rewards, rewards_s, plot_name):
    """Exact copy of show_progress_presi_gsi.py:plot_results."""
    iterations, final_rews, mean_rews = read_rewards(rewards)
    iterations_s, final_rews_s, mean_rews_s = read_rewards(rewards_s)

    fig, axs = plt.subplots(2, 1, sharex=True)

    ax = axs[0]
    ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    ax.plot(iterations, c=color)
    ax.plot(iterations_s, c=color, ls=':')
    ax.set_ylabel('steps', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax1 = plt.twinx(ax)
    color = 'k'
    ax1.plot(np.cumsum(iterations), c=color)
    ax1.plot(np.cumsum(iterations_s), c=color, ls=':')
    ax1.set_ylabel('cumulative steps', color=color)
    ax.set_title('Iterations')

    ax = axs[1]
    ax.axvspan(0, 100, alpha=0.2, color='coral')
    color = 'blue'
    ax.plot(mean_rews, c=color)
    ax.plot(mean_rews_s, c=color, ls=':')
    ax.set_ylabel('cum. return', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_title('Reward per episode')
    ax.set_xlabel('episodes')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.plot(final_rews[:-1], color=color)
    ax1.plot(final_rews_s[:-1], color=color, ls=':')
    ax1.set_ylabel('final return', color=color)
    ax1.axhline(-0.05, ls=':', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    fig.align_labels()
    fig.tight_layout()
    plt.savefig(os.path.join(FIG, plot_name + '_episodes.pdf'))
    plt.savefig(os.path.join(FIG, plot_name + '_episodes.png'))
    plt.close()


def read_losses_v_s(losses0, v_s0, max_length):
    """Exact copy of show_progress_presi_gsi.py:read_losses_v_s."""
    losses_all = []
    v_s_all = []
    for k in range(len(losses0)):
        losses = losses0[k]
        v_s = v_s0[k]
        losses_all.append(losses[:max_length])
        v_s_all.append(v_s[:max_length])
    losses = np.mean(losses_all, axis=0)
    v_s = np.mean(v_s_all, axis=0)
    return losses, v_s


def plot_convergence(losses, v_s, losses_s, v_s_s, label):
    """Based on show_progress_presi_gsi.py:plot_convergence.
    MODIFIED: Shows V only, no Bellman error (per user request)."""
    fig, ax = plt.subplots()
    ax.set_xlabel('# steps')

    color = 'lime'
    ax.set_ylabel('V', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(v_s, color=color)
    ax.plot(v_s_s, color=color, ls=':')

    plt.savefig(os.path.join(FIG, label + '_convergence.pdf'))
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# AE-DYNA — copied from AEdyna_clean_test.py:plot_observables (lines 1044-1091)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dyna_observables(data, label, out_name):
    """Exact copy of AEdyna_clean_test.py:plot_observables, adapted for save."""
    sim_rewards_all = np.array(data.get('sim_rewards_all'))
    step_counts_all = np.array(data.get('step_counts_all'))
    batch_rews_all = np.array(data.get('batch_rews_all'))
    tests_all = np.array(data.get('tests_all'))

    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.arange(len(batch_rews_all[0]))
    ax = axs[0]
    ax.step(x, batch_rews_all[0])
    ax.fill_between(x, batch_rews_all[0] - batch_rews_all[1], batch_rews_all[0] + batch_rews_all[1],
                    alpha=0.5)
    ax.set_ylabel('rews per batch')
    ax.set_title(label)

    ax2 = ax.twinx()
    color = 'lime'
    ax2.set_ylabel('data points', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.step(x, step_counts_all, color=color)

    ax = axs[1]
    ax.plot(sim_rewards_all[0], ls=':')
    ax.fill_between(x, sim_rewards_all[0] - sim_rewards_all[1], sim_rewards_all[0] + sim_rewards_all[1],
                    alpha=0.5)
    try:
        ax.plot(tests_all[0])
        ax.fill_between(x[:len(tests_all[0])],
                        tests_all[0] - tests_all[1], tests_all[0] + tests_all[1],
                        alpha=0.5)
        ax.axhline(y=np.max(tests_all[0]), c='orange')
    except:
        pass
    ax.set_ylabel('rewards tests')
    ax.grid(True)

    fig.align_labels()
    fig.tight_layout()
    plt.savefig(os.path.join(FIG, out_name + '.pdf'))
    plt.savefig(os.path.join(FIG, out_name + '.png'))
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ME-TRPO — copied from new_ME_TRPO_read_data_presi_gsi.py (lines 99-132)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_trpo_observables(data, out_name):
    """Exact copy of new_ME_TRPO_read_data_presi_gsi.py plotting code."""
    sim_rewards_all = data['sim_rewards_all'][0]
    entropy_all = data['entropy_all']
    step_counts_all = data['step_counts_all']
    batch_rews_all = data['batch_rews_all'][0]
    tests_all = data['tests_all'][0]

    fig, axs = plt.subplots(2, 1, sharex=True)
    x = np.arange(len(batch_rews_all))
    ax = axs[0]
    ax.step(x, batch_rews_all)
    ax.set_ylabel('rews per batch')

    ax2 = ax.twinx()
    color = 'lime'
    ax2.set_ylabel('data points', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.step(x, step_counts_all, color=color)

    ax = axs[1]
    ax.plot(sim_rewards_all, ls=':')
    ax.plot(tests_all)
    ax.set_ylabel('rewards model')

    ax2 = ax.twinx()
    color = 'lime'
    ax2.set_ylabel(r'- log(std($p_\pi$))', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.plot(entropy_all, color=color)

    fig.align_labels()
    fig.tight_layout()
    plt.savefig(os.path.join(FIG, out_name + '.pdf'))
    plt.savefig(os.path.join(FIG, out_name + '.png'))
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# DYNA verification — from new_ME_TRPO_read_data_presi_gsi.py:plot_results
# ═══════════════════════════════════════════════════════════════════════════════

def plot_verification(data, label, out_name):
    """Verification plot from new_ME_TRPO_read_data_presi_gsi.py:plot_results."""
    rewards = data

    iterations = []
    finals = []
    means = []

    for i in range(len(rewards)):
        if len(rewards[i]) > 1:
            finals.append(rewards[i][-1])
            means.append(np.sum(rewards[i][1:]))
            iterations.append(len(rewards[i]))

    fig, axs = plt.subplots(2, 1, sharex=True)

    ax = axs[0]
    ax.plot(iterations)
    ax.set_ylabel('steps')
    ax.set_title(label)

    ax = axs[1]
    color = 'blue'
    ax.set_ylabel('Final reward', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    ax.plot(finals, color=color)
    ax.set_title('Final reward per episode')
    ax.axhline(y=-0.05, c='blue', ls=':')
    ax.set_xlabel('Episodes (1)')

    ax1 = plt.twinx(ax)
    color = 'lime'
    ax1.set_ylabel('Cumulative reward', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.plot(means, color=color)

    fig.align_labels()
    fig.tight_layout()
    plt.savefig(os.path.join(FIG, out_name + '.pdf'))
    plt.savefig(os.path.join(FIG, out_name + '.png'))
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: Load data and reproduce all figures
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)

    # ── NAF Training + Verification ──────────────────
    print("NAF training + verification...")
    _, _, rewards_0, _ = load_pickle_logging('FEL_training_100_double_q_Tango_11')
    _, _, rewards_1, _ = load_pickle_logging('FEL_training_100_double_q_Tango_11_bis')
    rewards = [rewards_0, rewards_1]

    _, _, rewards_s0, _ = load_pickle_logging('FEL_training_100_single_q_Tango_11')
    _, _, rewards_s1, _ = load_pickle_logging('FEL_training_100_single_q_Tango_11_bis')
    rewards_s = [rewards_s1, rewards_s0]  # note: original had this order

    plot_results(rewards, rewards_s, 'FERMI_all_experiments_NAF')
    print("  ✓ FERMI_all_experiments_NAF_episodes.pdf")

    # ── NAF Convergence (V only) ─────────────────────
    print("NAF convergence (V only)...")
    rews0, inits0, losses0, v_s0 = load_pickle_final('FEL_training_100_double_q_Tango_11')
    rews1, inits1, losses1, v_s1 = load_pickle_final('FEL_training_100_double_q_Tango_11_bis')
    losses, v_s = read_losses_v_s([losses0, losses1], [v_s0, v_s1], 691)

    rews0, inits0, losses0, v_s0 = load_pickle_final('FEL_training_100_single_q_Tango_11')
    rews1, inits1, losses1, v_s1 = load_pickle_final('FEL_training_100_single_q_Tango_11_bis')
    losses_s, v_s_s = read_losses_v_s([losses0, losses1], [v_s0, v_s1], 691)

    plot_convergence(losses, v_s, losses_s, v_s_s, 'FERMI_all_experiments_NAF')
    print("  ✓ FERMI_all_experiments_NAF_convergence.pdf")

    # ── AE-DYNA Observables ──────────────────────────
    print("AE-DYNA observables...")
    # Find last training_observables in the DYNA directory tree
    for root, dirs, files in os.walk(DYNA):
        obs_files = sorted([f for f in files if 'training_observables' in f])
        if obs_files:
            with open(os.path.join(root, obs_files[-1]), 'rb') as f:
                dyna_obs = pickle.load(f)
            plot_dyna_observables(dyna_obs, 'AE-DYNA-SAC', 'AE-DYNA_observables')
            print("  ✓ AE-DYNA_observables.pdf")
            break

    # ── ME-TRPO Observables ──────────────────────────
    print("ME-TRPO observables...")
    # Original used run2
    trpo_run2 = os.path.join(TRPO, 'run2')
    obs_files = sorted([f for f in os.listdir(trpo_run2) if 'training_observables' in f])
    with open(os.path.join(trpo_run2, obs_files[-1]), 'rb') as f:
        trpo_obs = pickle.load(f)
    plot_trpo_observables(trpo_obs, 'ME-TRPO_observables')
    print("  ✓ ME-TRPO_observables.pdf")

    # ── DYNA Verification ────────────────────────────
    print("DYNA verification...")
    # AE-DYNA verification
    for root, dirs, files in os.walk(DYNA):
        finals = sorted([f for f in files if 'final' in f])
        if finals:
            with open(os.path.join(root, finals[-1]), 'rb') as f:
                dyna_final = pickle.load(f)
            plot_verification(dyna_final['rews'], 'AE-DYNA Verification', 'Verification_DYNA_all_episodes')
            print("  ✓ Verification_DYNA_all_episodes.pdf")
            break

    print("\n✓ All figures recreated from original scripts.")
