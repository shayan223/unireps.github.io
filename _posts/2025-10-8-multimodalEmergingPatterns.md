---
layout: distill
title: Understanding Adversarial Vulnerabilities and Emergent Patterns in Multimodal RL
description: Using a simplified multimodal RL agent to explore adversarial vulnerabilities that emerge when using different modalities.
tags: Adversarial Robustness, Reinforcement Learning, Multimodal models
giscus_comments: true
date: 2025-10-08
featured: true

authors:
  - name: Shayan Jalalipour
    url: "shayan2@pdx.edu"
    affiliations:
      name: Portland State University
  - name: Danielle Justo
    affiliations:
      name: Portland State University
  - name: Banafsheh Rekabdar
    affiliations:
      name: Portland State University

# Put a bib file for this post under assets/bibliography/
bibliography: 2025-10-08-multimodalEmergingPatterns.bib

toc:
  - name: Abstract
  - name: Introduction
  - name: Background
    subsections:
      - name: Related Works
      - name: Adversarial Attacks
      - name: Adversarial Defenses
      - name: Soft Actor Critic Models
  - name: Methodology
    subsections:
      - name: Agent
      - name: Environment
      - name: Adversarial Attack
      - name: Adversarial Defenses
      - name: Adversarial Evaluation
  - name: Results
    subsections:
      - name: Baseline Performance
      - name: Attacking an Undefended Model
      - name: Attacking a Defended Model
  - name: Conclusions
  - name: Reproducibility Details
---


## Introduction

Deep learning systems were once tailored to a single input type, for instance RGB pixels in image models <d-cite key="resnet2015"></d-cite> or tokenized text for language generation <d-cite key="Radford2018ImprovingLU"></d-cite>. Over the last few years we have watched machine learning expand toward richer multimodal setups that fuse information sources <d-cite key="jiao-multimodal-survey"></d-cite>. Teams now blend sensors on autonomous platforms <d-cite key="panduru2025exploring"></d-cite>, translate content across formats such as image to text or text to speech, and pair visual and language prompts for both large language models <d-cite key="openai2024gpt4technicalreport"></d-cite> and image generators <d-cite key="rombach2022highresolutionimagesynthesislatent"></d-cite>. Multimodal development is vibrant and fast moving.

With that momentum comes a pressing need to evaluate how these systems handle adversarial pressure. Every deep learning stack depends on trust, especially when it supports safety critical decisions. We must understand how attackers can exploit multimodal pipelines and how the interaction between modalities shapes new strengths and new points of failure. While single modality models have a long history of research on perturbations and data poisoning, the community still has only partial visibility into how those threats appear when modalities overlap.

Live reinforcement learning agents raise the stakes even further. These policies operate within dynamic environments rather than static classification tasks, so defenses must respect timing, feedback, and control constraints. They also learn without labeled supervision, which complicates how we adapt classic adversarial tooling.

Exploring the impact of these attacks on multimodal reinforcement learning matters because these agents power robotics and autonomous platforms. A brittle policy in those domains risks severe safety incidents and expensive hardware failures.

Our study aims to map the interplay between adversarial attacks, defense strategies, and modality combinations on a baseline multimodal reinforcement learning agent. We document baseline performance and walk through empirical findings that show how different modality pairings shift behavior when attacked jointly or separately. We highlight how influence varies by modality and how defensive choices reshape those dynamics.

To support this investigation we prepared datasets, trained reference models, and evaluated attack and defense mixes using a pipeline adapted from the open source DDiffPG release <d-cite key="li2024learningmultimodalbehaviorsscratch"></d-cite>.

Our contributions:
- Provide a testbed for adversarial evaluation of a multimodal reinforcement learning agent.
- Empirically characterize how attacking one or both modalities changes behavior.
- Show that defenses introduce emergent patterns across modalities, sometimes improving robustness and sometimes destabilizing training time adapted policies.


### Background and Related Works
Research on adversarial robustness has largely centered on language and vision systems, especially as large language models expanded into multimodal applications such as text to image and image to text experiences <d-cite key="wu2025dissectingadversarialrobustnessmultimodal,wang2025manipulatingmultimodalagentscrossmodal"></d-cite>. Investigations into decision making agents that blend multiple sensors remain comparatively sparse, with most of the attention directed toward autonomous driving stacks <d-cite key="chi_autonomous_survey2024,roheda2021multimodal"></d-cite>.

As embodied agents gain capability and broader deployment, we need a clearer view of how combined modalities influence security posture. Building resilient multimodal pipelines is essential for maintaining trust in systems that operate in high risk settings with steadily increasing task complexity.

### Adversarial Attacks
Adversarial attacks are a branch of machine learning focused on manipulating model behavior in unintended or harmful ways. In particular, adversarial attacks are aimed at a "victim" model often designed with the flaws of a particular type of model or architecture in mind. These attacks typically involve introducing carefully designed "perturbations" to the input, which are intended to mislead or alter the model's outputs. Depending on the attacker's level of access to the internals of the model, attacks are classified as "white box" (full access, such as weights or gradient values), "gray box" (partial access, such as particular values or weights) or "black box" (no access at all, only inputs and outputs can be discerned around the black box). While perturbing inputs is the most common form of adversarial attack, other methods such as dataset "Poisoning" exist which alter training data to induce some desired behavior from the victim model.

### Adversarial Defenses
Defenses against adversarial attacks have three main approaches: Augmenting training processes with adversarial samples <d-cite key="zizzo2021certified,tramer_adaptive_2020,kuzina_defending_2022"></d-cite>, detection of adversarial attacks, perturbations or anomalous data <d-cite key="roth_odds_2019,fidel_when_2020,guo_detecting_2019,GolchinAnomolyDetection"></d-cite>, and removing, filtering, or disrupting adversarial perturbations from the input data <d-cite key="zhang2021defense,nie_diffusion_2022,yoon_adversarial_2021"></d-cite>. For the purposes of our experiments, we utilize 3 methods for defense: Disruption, via gaussian noise; Detection, via neural network classifier and traditional clustering methods; Filtering, via Variational Autoencoder (VAE).

### Soft Actor Critic Models
Soft Actor Critic (SAC) <d-cite key="haarnoja2018softactorcritic"></d-cite> models are a Reinforcement Learning (RL) algorithm designed to solve unsupervised tasks such as embodying the Ant agent in our "Ant Maze" task. It uses an "Actor", a neural network acting as the agent policy, and "Critic" (or multiple critics) network as a value estimator. The key difference between a "Soft" and traditional Actor Critic is its use of an entropy term in its objective function, and the use of a stochastic policy with the intention of increasing training stability by encouraging exploration to avoid suboptimal convergence.

## Methodology

### Agent
To create a baseline agent, we train a Soft Actor Critic (SAC) agent <d-cite key="haarnoja2018softactorcritic"></d-cite> to embody the MuJoCo Ant <d-cite key="todorov2012mujoco,towers2024gymnasium"></d-cite>, a quadruped controlled by 8 rotors (with one positioned at each joint) with 4 limbs comprised of 2 "Links" each conjoined by a joint rotor.

The agent's observation space includes a velocity modality with linear and angular velocity using meters and radians per second, respectively. This includes velocities for all limbs and joints, including each limb link. Observations also include an angular modality in radians, tracking the angle between each link, the ant's torso orientation, as well as the angles of the limbs from the torso. Additionally, the model is supplied with a z coordinate torso reading. A position in meters representing the torso's height from the ground. Importantly, all observations are formally unbounded with a range of (−Inf, Inf).

The detailed observation space is outlined in the table below:

| Num | Observation | Name | Joint | Unit |
|-----|-------------|------|-------|------|
| 0 | z coordinate of the torso (centre) | torso | free | position (m) |
| 1 | x orientation of the torso (centre) | torso | free | angle (rad) |
| 2 | y orientation of the torso (centre) | torso | free | angle (rad) |
| 3 | z orientation of the torso (centre) | torso | free | angle (rad) |
| 4 | w orientation of the torso (centre) | torso | free | angle (rad) |
| 5 | angle between torso and front left link on front left | hip_1 (front_left_leg) | hinge | angle (rad) |
| 6 | angle between the two links on the front left | ankle_1 (front_left_leg) | hinge | angle (rad) |
| 7 | angle between torso and front right link on front right | hip_2 (front_right_leg) | hinge | angle (rad) |
| 8 | angle between the two links on the front right | ankle_2 (front_right_leg) | hinge | angle (rad) |
| 9 | angle between torso and back left link on back left | hip_3 (back_left_leg) | hinge | angle (rad) |
| 10 | angle between the two links on the back left | ankle_3 (back_left_leg) | hinge | angle (rad) |
| 11 | angle between torso and back right link on back right | hip_4 (right_back_leg) | hinge | angle (rad) |
| 12 | angle between the two links on the back right | ankle_4 (right_back_leg) | hinge | angle (rad) |
| 13 | x coordinate velocity of the torso | torso | free | velocity (m/s) |
| 14 | y coordinate velocity of the torso | torso | free | velocity (m/s) |
| 15 | z coordinate velocity of the torso | torso | free | velocity (m/s) |
| 16 | x coordinate angular velocity of the torso | torso | free | angular velocity (rad/s) |
| 17 | y coordinate angular velocity of the torso | torso | free | angular velocity (rad/s) |
| 18 | z coordinate angular velocity of the torso | torso | free | angular velocity (rad/s) |
| 19 | angular velocity of the angle between torso and front left link | hip_1 (front_left_leg) | hinge | angle (rad) |
| 20 | angular velocity of the angle between front left links | ankle_1 (front_left_leg) | hinge | angle (rad) |
| 21 | angular velocity of the angle between torso and front right link | hip_2 (front_right_leg) | hinge | angle (rad) |
| 22 | angular velocity of the angle between front right links | ankle_2 (front_right_leg) | hinge | angle (rad) |
| 23 | angular velocity of the angle between torso and back left link | hip_3 (back_left_leg) | hinge | angle (rad) |
| 24 | angular velocity of the angle between back left links | ankle_3 (back_left_leg) | hinge | angle (rad) |
| 25 | angular velocity of the angle between torso and back right link | hip_4 (right_back_leg) | hinge | angle (rad) |
| 26 | angular velocity of the angle between back right links | ankle_4 (right_back_leg) | hinge | angle (rad) |

*Table: Observations space for the MuJoCo Ant Maze task. Ranges for all values are −Inf to +Inf.*

The action space consists of 8 torque values applied to the rotors:

| Num | Action | Name | Joint | Type (Unit) |
|-----|--------|------|-------|-------------|
| 0 | Torque applied on the rotor between the torso and back right hip | hip_4 (right_back_leg) | hinge | torque (N m) |
| 1 | Torque applied on the rotor between the back right two links | angle_4 (right_back_leg) | hinge | torque (N m) |
| 2 | Torque applied on the rotor between the torso and front left hip | hip_1 (front_left_leg) | hinge | torque (N m) |
| 3 | Torque applied on the rotor between the front left two links | angle_1 (front_left_leg) | hinge | torque (N m) |
| 4 | Torque applied on the rotor between the torso and front right hip | hip_2 (front_right_leg) | hinge | torque (N m) |
| 5 | Torque applied on the rotor between the front right two links | angle_2 (front_right_leg) | hinge | torque (N m) |
| 6 | Torque applied on the rotor between the torso and back left hip | hip_3 (back_leg) | hinge | torque (N m) |
| 7 | Torque applied on the rotor between the back left two links | angle_3 (back_leg) | hinge | torque (N m) |

*Table: MuJoCo Ant agent action space. All values range between [−1, 1].*

Below is an illustration of the body and rotor layout used in our experiments:

<div class="l-body-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/example_ant.png' | relative_url }}" alt="Example Mujoco Ant body." style="width: 100%; max-width: 300px;" />
        <figcaption>Example MuJoCo Ant body <d-cite key="towers2024gymnasium"></d-cite>.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/ant_joints.png' | relative_url }}" alt="Rotor layout for Ant." style="width: 100%; max-width: 300px;" />
        <figcaption>Rotor layout for Ant <d-cite key="towers2024gymnasium"></d-cite>.</figcaption>
      </figure>
    </div>
  </div>
</div>

### Environment
In addition to training the agent to embody its quadruped ant, we train it for the AI Gymnasium task "Ant Maze" <d-cite key="towers2024gymnasium"></d-cite>. This task requires the agent to learn how to walk with its body then utilize its learned movements to maneuver around obstacles to reach a predetermined destination. This can be seen in pathing and exploration heat map figures throughout this paper.

### Adversarial Attack
We rely on the Fast Gradient Sign Method (FGSM) <d-cite key="goodfellow2015explaining"></d-cite> as the baseline attack across our experiments. FGSM is a white box technique that perturbs inputs using the sign of the gradient from the victim model. The standard form is

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon\,\mathrm{sign}\big(\nabla_{\mathbf{x}} J(\theta,\mathbf{x},y)\big)
$$

Here $\mathbf{x}$ is the original input, $y$ is the true label, $\epsilon$ scales the perturbation, $J(\theta,\mathbf{x},y)$ is the loss, $\nabla_{\mathbf{x}} J(\theta,\mathbf{x},y)$ is the gradient with respect to the input, $\mathrm{sign}(\cdot)$ is the element wise sign function, and $\mathbf{x}_{\text{adv}}$ is the crafted adversarial example.

We make two adjustments relative to the original presentation. FGSM usually targets image classifiers and displays perturbations as pixel shifts. Our experiments work directly with the observation vector, so we craft perturbations in that space using the appropriate units for each modality, such as radians or meters per second.

FGSM also assumes labeled data to compute a loss. Reinforcement learning policies operate without labels, so we adapt the objective to reflect the critic value instead. Our targeted degradation objective becomes

$$
\mathbf{x}_{\text{adv}} = \mathbf{x} + \epsilon\,\mathrm{sign}\big(\nabla_{\mathbf{x}} Q_{\theta}(\mathbf{x}, a)\big)
$$

This change draws on the Q function defined by the critic in our actor critic setup rather than a loss based on class labels. It lets FGSM act directly on the live policy by following the Q network assessment of each action when computing perturbations.

### Adversarial Defenses
Our evaluation includes three defense themes. We start with a disruption baseline that applies scaled gaussian noise to the observation vector. We then explore adversarial detection by training classifiers to flag perturbed observations and compare them with traditional clustering tools such as K Means and Gaussian Mixture Models (GMM). Finally, we assess a purification pipeline built on a defense VAE that reconstructs benign versions of the inputs.

Both the neural network detectors and the clustering approaches rely on a dataset collected during SAC training and evaluation. The agent runs for three million steps. We treat the first one point five million steps as a warm up phase to avoid logging data from an agent that has not yet learned to move reliably. During the final one point five million steps we record observations into a benign dataset. We also generate a matching adversarial dataset by applying FGSM perturbations to those observations without feeding the altered signals back to the agent. The result is a paired corpus of benign and adversarial samples for every modality.

#### Gaussian Noise Defense
The gaussian noise filter serves as the baseline defense. We sample perturbations from a normal distribution and scale them by $\epsilon$:

$$
\mathbf{x_{def}} = x + \epsilon \cdot \mathbf{n}, \quad \text{where } \mathbf{n} \sim \mathcal{N}(0, \mathbf{I})
$$

Targeted adversarial perturbations are often sensitive to small changes. By injecting a modest level of noise we can disrupt their structure and blunt the attack, accepting some degradation from the added randomness. This approach is simple and computationally inexpensive.

#### Defense VAE
Our defense VAE follows prior work on variational autoencoder purification <d-cite key="li2019defensevaefastaccuratedefense,shayanNDVAE"></d-cite>. The model trains on paired benign and adversarial samples like those described above. The encoder observes both forms, while the decoder learns to reconstruct the benign target. Over time the encode decode pathway maps adversarial inputs back into the benign observation space.

We adapt the architecture from image defenses to match our one dimensional observation vector. The compact network uses four fully connected layers with ReLU activation rather than convolutional stacks. Since the modalities are not arranged as a sequence with positional structure, we also avoid one dimensional convolutions.

#### Adversarial Detection
Detection centric defenses focus on identifying an attack rather than intervening directly in the control loop. We therefore evaluate them on prediction accuracy and F1 score but do not alter the agent mid run. A production system could take many actions once an attack is detected, yet that follow up is outside the scope of this study.

The labeled dataset lets us train Support Vector Machines (SVM), K Nearest Neighbors, and neural network classifiers to distinguish benign from adversarial observations. For comparison we also fit simpler clustering approaches such as K means and Gaussian Mixture Models to the same data.

Classifier accuracy reflects binary predictions on benign versus adversarial inputs. For the unsupervised clustering methods we assign cluster labels to maximize accuracy after fitting the two cluster model.

### Adversarial Evaluation
To test the effects of adversarial attacks on our baseline agent across its modalities, we use the following process:

1. The agent is trained normally on benign inputs.
2. Every evaluation episode, the attack is applied to perturb the input: First across both modalities simultaneously, then individually by modifying only the target modality values in the agent's observation vector.
3. We compare benign training performance and adversarial evaluation to identify effects of the attack on the agent.

To develop a thorough understanding of model performance when defended, each defense method is applied in two configurations: In the first configuration, the model trains as normal, and the defense method is applied during evaluation, prior to the attacked data reaching the model. In the second configuration, the model also trains on benign inputs that have first been given to the defense method. The model is then attacked as usual. This helps compare the effects of model tuning on each defense.

## Results

### Baseline Performance
We begin by training an SAC agent on the Ant Maze task for three million steps so that it reliably clears a simple obstacle. The configuration mirrors common SAC settings with a replay buffer of one million transitions, $\tau=0.05$, and $\gamma=0.99$.

The figures below trace the training story. Rewards rise and level off after the run, the exploration heatmap shows how the agent learns an efficient route, and evaluation rewards peak once the policy stabilizes. The path visual confirms that the agent reaches the goal consistently.

With no adversarial pressure this baseline agent handles the maze it was trained on.

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_train.png' | relative_url }}" alt="SAC training reward." style="width: 100%; max-width: 400px;" />
        <figcaption>Training reward progression (3M steps).</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_explore.png' | relative_url }}" alt="Exploration heatmap." style="width: 100%; max-width: 400px;" />
        <figcaption>Exploration heatmap during learning.</figcaption>
      </figure>
    </div>
  </div>
</div>

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_eval.png' | relative_url }}" alt="Evaluation reward." style="width: 100%; max-width: 400px;" />
        <figcaption>Evaluation performance plateaus after training.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/baseline_paths.png' | relative_url }}" alt="Paths to goal." style="width: 100%; max-width: 400px;" />
        <figcaption>Typical paths from start to goal after training.</figcaption>
      </figure>
    </div>
  </div>
</div>

### Attacking an Undefended Model
With the modified FGSM attack ($\epsilon=0.005$) we perturb the observation vector during evaluation. The attack seeks the lowest value outcome predicted by the critic, pushing the policy toward poor decisions.

#### Attacking Both Modalities
When we perturb both modalities, reward traces reveal the story immediately. Sharp drops in the purple line show successful attacks, while quick recoveries indicate attempts that did not hold. The path comparisons illustrate what those swings look like in the environment: a failed attack nudges the agent onto an alternate route, whereas a successful one leaves the agent wandering near the start.

#### Attacking Individual Modalities
We then target each modality on its own. The performance plot highlights how velocity (green) and angle (blue) perturbations differ from the full multimodal attack. Velocity covers roughly one quarter of the observation vector, so those attacks fail more often and produce taller reward spikes. Angle perturbations sit between the two extremes.

These results reinforce a practical takeaway: the share of the observation space tied to a modality limits how much damage an attacker can cause when only that modality is manipulated.

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/fgsm_eval.png' | relative_url }}" alt="FGSM evaluation performance." style="width: 100%; max-width: 600px;" />
  <figcaption>Benign (red) vs adversarial (purple) evaluation. Sharp reward drops indicate successful attacks.</figcaption>
</figure>

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/fgsm_path_success.png' | relative_url }}" alt="FGSM success example." style="width: 100%; max-width: 350px;" />
        <figcaption>FGSM attack success: agent gets lost early.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/fgsm_path_fail.png' | relative_url }}" alt="FGSM failure example." style="width: 100%; max-width: 350px;" />
        <figcaption>FGSM attack failure: minimal route alteration.</figcaption>
      </figure>
</div>
</div>
</div>

Modality specific FGSM shows that attack effectiveness scales with the attacked fraction of the observation space.

<figure class="l-page">
  <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/individual_modality_fgsm.png' | relative_url }}" alt="Modality specific FGSM results." style="width: 100%; max-width: 600px;" />
  <figcaption>Performance under FGSM for both modalities (purple), angles only (blue), and velocity only (green).</figcaption>
</figure>

### Attacking a Defended Model
With the baseline established, we introduce defenses to see how they reshape the interaction between attacker and agent.

#### Gaussian Noise Defense
The gaussian noise filter is the simplest option. We add noise drawn from a normal distribution with mean zero and standard deviation one, scaled by 0.005.

Even this basic disruption helps. Both evaluation only noise (red) and noise seen during training (orange) outperform the undefended purple curve. Injecting randomness breaks up many attacks.

Targeting individual modalities adds an unexpected twist. Velocity attacks behave much like the multimodal case, but angle attacks reveal that training with noise can actually destabilize performance. The best outcome there comes from applying noise only at evaluation time, a pattern we observed across repeated runs.

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/noise_defense_fgsm.png' | relative_url }}" alt="Noise defense under multimodal FGSM." style="width: 100%; max-width: 400px;" />
        <figcaption>Defense only (red), trained with noise (orange), undefended (purple).</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/angular_noise_defense_fgsm.png' | relative_url }}" alt="Noise defense for angle only attacks." style="width: 100%; max-width: 400px;" />
        <figcaption>Angle only attacks: defense at eval (purple) outperforms training through noise (green) and baseline (blue).</figcaption>
      </figure>
    </div>
  </div>
</div>

#### Defense VAE
The compact Defense VAE offers another perspective. Running it only during evaluation lifts rewards under FGSM, but training the agent on VAE filtered inputs prevents the policy from solving the task. Single modality attacks expose its weaknesses: the VAE performs best when both modalities are perturbed together and struggles with narrow attacks.

This pattern lines up with how the VAE learned. Training on paired benign and adversarial samples covering the entire observation space primes it to recognize broad perturbations. Once the attacker focuses on a single modality the reconstruction no longer matches the training distribution, so protection fades.

<div class="l-page-outset">
  <div class="row">
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/vae_both_modalities.png' | relative_url }}" alt="VAE defense under multimodal FGSM." style="width: 100%; max-width: 400px;" />
        <figcaption>VAE defense (blue) vs undefended adversarial inputs (purple) at $\epsilon=0.007$.</figcaption>
      </figure>
    </div>
    <div class="col-sm-6">
      <figure>
        <img src="{{ '/assets/img/2025-10-08-multimodalEmergingPatterns/vae_individual_modalities.png' | relative_url }}" alt="VAE defense under modality specific FGSM." style="width: 100%; max-width: 400px;" />
        <figcaption>VAE under both modalities (blue), velocity only (red), angle only (orange) attacks.</figcaption>
      </figure>
    </div>
  </div>
</div>

#### Detection Results (Summary)
The classifier roundup shows a clear trend: higher $\epsilon$ values make detection easier. Angular perturbations stand out at $\epsilon=0.007$, while velocity attacks become more visible at $\epsilon=0.015$. Clustering baselines such as K means and GMM hover near chance overall.

Clustering does show a small edge when angles are perturbed, hinting that those signals are easier to separate without labels. The broader lesson is that modality combinations shift detection difficulty, and scaling the attack changes that balance even when success rates stay similar.

The detailed results for all detection methods are shown in the table below:

| Detection Method | Multimodal | | | Velocity | | | Angular | | |
|------------------|-------------|---|---|----------|---|---|---------|---|---|
| | Accuracy | F1 Score | Epsilon | Accuracy | F1 Score | Epsilon | Accuracy | F1 Score | Epsilon |
| SVM | 0.6027 | 0.6936 | 0.007 | 0.4993 | 0.6227 | 0.007 | 0.5606 | **0.6596** | 0.007 |
| SVM | 0.7188 | 0.7608 | 0.015 | 0.6070 | 0.7012 | 0.015 | 0.5048 | 0.3782 | 0.015 |
| KNN | 0.5909 | 0.5897 | 0.007 | 0.5164 | 0.4718 | 0.007 | 0.5391 | 0.5412 | 0.007 |
| KNN | 0.6606 | 0.6522 | 0.015 | 0.5850 | 0.5769 | 0.015 | 0.5152 | 0.3962 | 0.015 |
| NN | 0.7349 | 0.725 | 0.007 | 0.5494 | 0.6437 | 0.007 | 0.7104 | 0.7372 | 0.007 |
| NN | **0.9892** | **0.9892** | 0.015 | **0.8766** | **0.8810** | 0.015 | **0.8211** | 0.8264 | 0.015 |
| GMM | 0.4989 | N/A | 0.007 | 0.4984 | N/A | 0.007 | 0.4996 | N/A | 0.007 |
| GMM | 0.5094 | N/A | 0.015 | 0.4976 | N/A | 0.015 | 0.5021 | N/A | 0.015 |
| Kmeans | 0.5033 | N/A | 0.007 | 0.4998 | N/A | 0.007 | 0.5026 | N/A | 0.007 |
| Kmeans | 0.5416 | N/A | 0.015 | 0.4602 | N/A | 0.015 | 0.4855 | N/A | 0.015 |

*Table: Classifier performance across modalities with FGSM epsilon (scaling) values of 0.007 and 0.015.*

## Conclusions
Our experiments confirm that a multimodal agent can be pushed off course through attacks on any of its inputs. The larger a modality’s footprint in the observation vector, the more influence an attacker gains. Defenses shape those dynamics in different ways: gaussian noise offers broad value with modality specific caveats, and the Defense VAE excels when the attack distribution matches its training data.

These findings underline the importance of viewing reinforcement learning robustness through a modality aware lens. Simple interventions already reveal nuanced behavior, and the pipeline we release provides a starting point for deeper explorations.

The baseline setup here is intentionally lightweight, yet it highlights clear research directions for more complex agents and environments. We hope the insights and tools accelerate progress on safeguarding the next wave of multimodal systems.

## Reproducibility Details
Experiments used SAC with 3M steps, replay size 1e6, $\tau=0.05$, $\gamma=0.99$, evaluation every 100 steps, and Adam learning rates as in our code. Defense VAE used fully connected enc/dec layers (256-128-64 latent-64-128-256) with latent size 24, trained for 50 epochs. Classifier details and additional hyperparameters are available in the appendix of the paper and our codebase.

### SAC Hyperparameters

| Parameter | Value |
|-----------|-------|
| Horizon Length | 1 |
| Memory Size | $1 \times 10^6$ |
| Batch Size | 4096 |
| $N$-step | 1 |
| $\tau$ (Target smoothing coefficient) | 0.05 |
| $\gamma$ (Discount factor) | 0.99 |
| Warm up Steps | 32 |
| Critic Class | Double Q Network |
| Evaluation Frequency | 100 |
| Learning Rate (Alpha) | 0.005 |
| Update Times per Step | 8 |

### VAE Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | $1 \times 10^{-3}$ |
| Batch Size | 32 |
| Training Epochs | 50 |
| Latent Dimension | 24 |
| Encoder Layers | 5 (Observation, 256, 128, 64, Latent Dimension) |
| Decoder Layers | 5 (Latent Dimension, 64, 128, 256, Observation) |

### Neural Network Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input Dimension | 28 |
| Output Dimension | 1 |
| Hidden Layers | 5 (256, 256, 128, 32, 1) |
| Learning Rate | 0.0001 |
| Training Epochs | 60 |

### Support Vector Machine (SVM) Hyperparameters

| Parameter | Value |
|-----------|-------|
| $C$ | 1000 |
| Gamma | 1 |
| Degree | 3 |
| Decision Function | One vs One (ovo) |

### GMM Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of Components | 2 |
| Initialization Method | k means++ |
| Covariance Type | full |
| Convergence Tolerance ($\texttt{tol}$) | 0.001 |
| Regularization of Covariance ($\texttt{reg\_covar}$) | $1 \times 10^{-6}$ |
| Max Iterations | 100 |
| Number of Initializations ($\texttt{n\_init}$) | 1 |

### K Means
K Means clustering algorithm as implemented by Scikit learn, with a k value of 2 clusters.

<d-appendix>
  <d-footnote>
    We release a lightweight pipeline adapted from DDiffPG <d-cite key="li2024learningmultimodalbehaviorsscratch"></d-cite> to reproduce training, dataset generation, attacks, and defenses.
  </d-footnote>
</d-appendix>
