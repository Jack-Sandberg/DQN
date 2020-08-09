DQN
================

In this project, I have implemented the Deep Q Network presented by
DeepMind researchers in [the original
paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and follow-up
[Nature
paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
I have also implemented architectures that improve upon the original
DQN: [Double DQN](https://arxiv.org/abs/1509.06461) and [Dueling
DQN](https://arxiv.org/abs/1511.05952).

### Evaluation samples from Dueling Double DQN with Prioritized Replay Buffer:

|                                   Episode 50                                    |                                    Episode 150                                    |                                    Episode 750                                     |
| :-----------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| <img src = "Pong/DuelingDoubleDQNGifs/eval_ep_50_step_51540.gif" width = 200 /> | <img src = "Pong/DuelingDoubleDQNGifs/eval_ep_150_step_312995.gif" width = 200 /> | <img src = "Pong/DuelingDoubleDQNGifs/eval_ep_750_step_1730869.gif" width = 200 /> |

### Reward and Q-value plots

<img src="readme_files/figure-gfm/all plots-1.svg" style="display: block; margin: auto;" />
The first two figures show that all models learn to solve Pong with D3QN
with Prio and DQN being the faster learners. The third figure shows that
DQN initially learns that the average Q-value is negative while playing
randomly. After some time, the average Q-values increase along with the
rewards received. The fourth figure shows that the maximum Q-value
calculated per episode increases steadily at the start for D3Qn and DQN.
In general, the max Q-value is significantly larger than the average
Q-value which shows that DQN can distinguish between states of different
values.

Keep in mind that the learning speed of DQN can be sensitive to the
random seed assigned and that the models were only trained once.
Additionally, the proposed improvements to DQN (Double, Dueling and PER)
were assessed on the scores across on all Atari games and not on Pong in
particular.

### Resources used

This project would not have been possible without all the brilliant
resources available on the Internet. Among many, the following resources
have helped me the most:

  - [DQN tutorial for
    PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
      - Introductory tutorial to DQN that got me off the ground.
  - [OpenAI Baselines](https://github.com/openai/baselines)
      - Provides implementations of most Deep Reinforcement Learning
        algorithms written in Tensorflow. However, I used Baselines
        [ReplayBuffer](https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py)
        to create ReplayBufferTorch and the wrappers available
        [here](https://github.com/openai/baselines/tree/master/baselines/common).
  - [Pre-processing of
    frames](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/)
      - Goes through the pre-processing Deepmind used in their DQN paper
        in great detail.
  - [Guide to speeding up
    DQN](https://medium.com/@shmuma/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55)
      - Excellent guide to speeding up the convergence of DQN, provides
        hyperparameters that converges faster.

### Hyperparameters

Trained for \~800 episodes and performed an evaluation every 50 episodes
that consisted of playing 5 episodes.

  - Update frequency = 4 (number of steps in the environment before
    performing an optimization step),
  - Batch size = 32\*4 = 128,
  - Gamma = 0.99,
  - Epsilon linearly decreasing from 1 to 0.02 over 100 000 steps with
    an evaluation epsilon of 0.002,
  - Target network updated every 1 000 steps,
  - Replay buffer initialized with 10 000 random steps and maximum
    capacity of 100 000,
  - Learning rate for ADAM optimizer = 0.0001,
  - Prioritized Experience Replay:
      - Alpha = 0.6,
      - Beta linearly increasing from 0.4 to 1 over 20 000 000 steps.
