# Online Model-Based and Model-Free Reinforcement Learning in Accelerator Operation with Applications to FERMI FEL

## These are the results of RL tests @FERMI-FEL
The problem has four degrees of freedom in state and action space.
A schematic overview:

![SchemaFERMIFEL](Figures/SL_Alignment_Scheme.png)

Algorithm | Type | Representational power|Noise resistive|Sample efficiency
------------ | -------------|---------|------------|---------
_NAF_ | Model-free|Low|No|High
_NAF2_ | Model-free|Low|Yes|High
_ME-TRPO_ | Model-based|High|No|High
_AE-DYNA_ | Model-based|High|Yes|High

## Experiments done on the machine:

A new implementation of the NAF with doule Q learning:

![NAF2_training](Figures/FERMI_all_experiments_NAF_episodes.png)

![NAF2_training](Figures/FERMI_all_experiments_NAF_convergence.png)

A new implementation of a _AE-DYNA_:

![AE-DYNA](Figures/AE-DYNA_observables.png)

![AE-DYNA](Figures/AE-DYNA_verification.png)

A variant of the _ME-TRPO_:

![ME-TRPO](Figures/ME-TRPO_observables.png)

![ME-TRPO](Figures/ME-TRPO_verification.png)

## The evolution as presented at GSI [Towards Artificial Intelligence in Accelerator Operation](https://indico.gsi.de/event/11539/):
![ME-TRPO](Figures/Learning_evolution.png)

## Experiments done on the [_inverted pendulum_](https://gym.openai.com/envs/Pendulum-v0/) openai gym environment:

Cumulative reward of different _NAF_ implementations on the _inverted pendulum_ with artificial noise.

![NAF_NOISE](Figures/Comparison_noise.png)

Comparison of the inclusion of aleatoric noise in the AE-DYNA in the noisy _inverted pendulum_:

![AE-DYNA_NOISE](Figures/Comparison_noise_ae_dyna.png)

Comparison of the inclusion of aleatoric noise in the AE-DYNA in the noisy _inverted pendulum_:

![AE-DYNA_NOISE](Figures/Comparison_models_sizes.png)

Sample efficiency of _NAF_ and _AE-DYNA_:

![AE-DYNA](Figures/Comparison_NAF_AE-DYNA.png)

Free run on the _inverted pendulum_:

![AE-DYNA](Figures/AE-DYNA_free_run_pendulum.png)

