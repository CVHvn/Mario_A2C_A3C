# Mario_A2C_A3C_ACKTR
Playing Super Mario Bros with Advantage Actor Critic (A2C), Asynchronous Advantage Actor Critic (A3C) and Actor Critic using Kronecker-Factored Trust Region (ACKTR)

## Introduction

My PyTorch [Advantage Actor Critic (A2C) is the synchronous version of the A3C](https://arxiv.org/pdf/1602.01783), [Asynchronous Advantage Actor Critic (A3C)](https://arxiv.org/pdf/1602.01783) and [Actor Critic using Kronecker-Factored Trust Region (ACKTR)](https://arxiv.org/pdf/1708.05144) implement to playing Super Mario Bros.

This is A2C result:
<p align="center">
  <img src="A2C/demo/gif/1-1.gif" width="200">
  <img src="A2C/demo/gif/1-2.gif" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200">
  <img src="A2C/demo/gif/1-4.gif" width="200"><br/>
  <img src="A2C/demo/gif/2-1.gif" width="200">
  <img src="A2C/demo/gif/2-2.gif" width="200">
  <img src="A2C/demo/gif/2-3.gif" width="200">
  <img src="A2C/demo/gif/2-4.gif" width="200"><br/>
  <img src="A2C/demo/gif/3-1.gif" width="200">
  <img src="A2C/demo/gif/3-2.gif" width="200">
  <img src="A2C/demo/gif/3-3.gif" width="200">
  <img src="A2C/demo/gif/3-4.gif" width="200"><br/>
  <img src="A2C/demo/gif/4-1.gif" width="200">
  <img src="A2C/demo/gif/4-2.gif" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200"><br/>
  <img src="A2C/demo/gif/5-1.gif" width="200">
  <img src="A2C/demo/gif/5-2.gif" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200">
  <img src="A2C/demo/gif/5-4.gif" width="200"><br/>
  <img src="A2C/demo/gif/6-1.gif" width="200">
  <img src="A2C/demo/gif/6-2.gif" width="200">
  <img src="A2C/demo/gif/6-3.gif" width="200">
  <img src="A2C/demo/gif/6-4.gif" width="200"><br/>
  <img src="A2C/demo/gif/7-1.gif" width="200">
  <img src="A2C/demo/gif/7-2.gif" width="200">
  <img src="A2C/demo/gif/7-3.gif" width="200">
  <img src="A2C/demo/gif/7-4.gif" width="200"><br/>
  <img src="A2C/demo/gif/8-1.gif" width="200">
  <img src="A2C/demo/gif/8-2.gif" width="200">
  <img src="A2C/demo/gif/8-3.gif" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200"><br/>
  <i>A2C Results</i>
</p>

This is A3C result (can't complete 1-3, 4-3, 4-4, 5-3, 8-4. Other stages I don't have resources to try):

<p align="center">
  <img src="A3C/demo/gif/1-1.gif" width="200">
  <img src="A3C/demo/gif/1-2.gif" width="200">
  <img src="A3C/demo/gif/1-4.gif" width="200">
  <img src="A3C/demo/gif/2-2.gif" width="200"><br/>
  <i>A3C Results</i>
</p>

This is ACKTR result:
<p align="center">
  <img src="ACKTR/demo/gif/1-1.gif" width="200">
  <img src="ACKTR/demo/gif/1-2.gif" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200">
  <img src="ACKTR/demo/gif/1-4.gif" width="200"><br/>
  <img src="ACKTR/demo/gif/2-1.gif" width="200">
  <img src="ACKTR/demo/gif/2-2.gif" width="200">
  <img src="ACKTR/demo/gif/2-3.gif" width="200">
  <img src="ACKTR/demo/gif/2-4.gif" width="200"><br/>
  <img src="ACKTR/demo/gif/3-1.gif" width="200">
  <img src="ACKTR/demo/gif/3-2.gif" width="200">
  <img src="ACKTR/demo/gif/3-3.gif" width="200">
  <img src="ACKTR/demo/gif/3-4.gif" width="200"><br/>
  <img src="ACKTR/demo/gif/4-1.gif" width="200">
  <img src="ACKTR/demo/gif/4-2.gif" width="200">
  <img src="ACKTR/demo/gif/4-3.gif" height="187.5" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200"><br/>
  <img src="ACKTR/demo/gif/5-1.gif" width="200">
  <img src="ACKTR/demo/gif/5-2.gif" width="200">
  <img src="ACKTR/demo/gif/5-3.gif" height="187.5" width="200">
  <img src="ACKTR/demo/gif/5-4.gif" width="200"><br/>
  <img src="ACKTR/demo/gif/6-1.gif" width="200">
  <img src="ACKTR/demo/gif/6-2.gif" width="200">
  <img src="ACKTR/demo/gif/6-3.gif" width="200">
  <img src="ACKTR/demo/gif/6-4.gif" width="200"><br/>
  <img src="ACKTR/demo/gif/7-1.gif" width="200">
  <img src="ACKTR/demo/gif/7-2.gif" width="200">
  <img src="ACKTR/demo/gif/7-3.gif" width="200">
  <img src="ACKTR/demo/gif/7-4.gif" width="200"><br/>
  <img src="ACKTR/demo/gif/8-1.gif" width="200">
  <img src="ACKTR/demo/gif/8-2.gif" width="200">
  <img src="ACKTR/demo/gif/8-3.gif" width="200">
  <img src="A2C/demo/Black_colour.jpg" height="187.5" width="200"><br/>
  <i>ACKTR Results</i>
</p>

## Motivation

I've been interested in Reinforcement Learning since I was in university, but I only tested Atari games with the recommended hyperparameters. If you've ever studied RL, you'll realize that finding the right hyperparameters is extremely important for most RL algorithms (because RL is often very sensitive to hyperparameters). Now I want to try using RL to train the agent to play another game instead of Atari like in the papers.

Initially, I only intended to use the stable baseline to train the agent with some lines of code. But I realized that the documentation and code were very hard to read and it was difficult for me to adjust the code.

I have reviewed many other source codes to train agents to play Mario, but most of them only code demos and complete a few very easy states like 1-1, 1-4 and sometimes have some bugs in the code. They also often use simple or right only action_space to make the agent learning easier. So I decided to implement A2C so I can easily adjust the code and understand the algorithm more deeply. I tried tuning the hyperparameters to solve as many stages as possible with this source code.

After completed A2C, I want to try stronger algorithm like PPO, A3C, ACKTR. Because A3C, ACKTR have many similarities with A2C and I don't want create many repos with same content (play mario with weak algorithms), I don't seperate A3C and ACKTR to new repos. 

## How to use it

### A2C

You can use my [A2C notebook](a2c_lstm_mario.ipynb) for training and testing agent very easy:
* **Train your model** by running all cell before session test
* **Test your trained model** by running all cell except agent.train(), just pass your model path to agent.load_model(model_path)

Or you can use [**A2C/train.py**](A2C/train.py) and [**A2C/test.py**](A2C/test.py) if you don't want to use notebook:
* **Train your model** by running **A2C/train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model** by running **A2C/test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth  --num_envs 2

### A3C

Use [**A3C/train.py**](A3C/train.py) and [**A3C/test.py**](A3C/test.py):
* **Train your model** by running **A3C/train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 8
* **Test your trained model** by running **A3C/test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth

A3C use multi process make it hard to work with notebook --> I don't use notebook for A3C version!

### ACKTR

You can use my [ACKTR notebook](acktr_mario.ipynb) for training and testing agent very easy:
* **Train your model** by running all cell before session test
* **Test your trained model** by running all cell except agent.train(), just pass your model path to agent.load_model(model_path)

Or you can use [**ACKTR/train.py**](ACKTR/train.py) and [**ACKTR/test.py**](ACKTR/test.py) if you don't want to use notebook:
* **Train your model** by running **ACKTR/train.py**: For example training for stage 1-4: python train.py --world 1 --stage 4 --num_envs 32
* **Test your trained model** by running **ACKTR/test.py**: For example testing for stage 1-4: python test.py --world 1 --stage 4 --pretrained_model best_model.pth

## Trained models

You can find A2C trained model in folder [trained_model](A2C/trained_model).

You can find A3C trained model in folder [trained_model](A3C/trained_model).

You can find ACKTR trained model in folder [trained_model](ACKTR/trained_model).

## Hyperparameters

### A2C

I use hyperparameters as this table to train agent. How I find hyperparameters:
* First, I find default hyperparameters from other implements and Atari hyperparameters like: learning rate is 1e-4, gamma is 0.9, learn_step is 20 and num_envs is 16 (for stages 7-1, 7-2, 7-3, I just use num_envs is 8 because I don't have enought resource, set num_envs to 16 still better for this stages).
* Try default hyperparameters.
* Tune hyperparameters if agent failed. I found that only gamma and learn_step help agent learn better.
* Changing the learning rate does not improve the agent and we cannot find the optimal learning rate except by trial and error (consuming a lot of resources).
* For some stages that require a long sequence of actions (like jumping over a deep hole), setting a gamma of 0.99 helps the model complete the stage while 0.9 cannot.
* Setting learn_step to 5 usually helps the model learn better (especially helps complete stage 8-1) but training time and stability will decrease compared to learn_step of 20.

**Update**: 
- LSTM backpropagation through time: I find that my code have mistake (I detach lstm hidden state in model.predict() than my model just use h, c as inputs). To take advantage of lstm's backpropagation through time, I fix model by not detach h, c.
  - Because after backward, we need release gradient in h, c (or bug), I need detach h, c after each training step.
  - Because my old config (detach_lstm_state = True, don't backpropagation through time) still work (complete 26/32 stages). I add hyperparameter detach_lstm_state to setup whether to use backpropagation through time or not.
  - Backpropagation through time help me complete state 6-3 (my old version can't complete this stages). Note: You will complete this stage with init_weights = True.
- init_weights: in my PPO project (I done it later). I find that we don't need init weight with A2C/A3C/PPO model. Than I add hyperparameter init_weights.
- adam_eps: I complete 26/32 stages with default PyTorch eps (1e-8). But I find that many projects and papers recomment use 1e-5 or larger eps. Than I add hyperparameter adam_eps.
  - With my knowledge, if you want increase eps (1e-5 or even 0.1), you need increase learning rate.
  - I found increasing eps did not help the algorithm better with my tests. If you want to improve, spend time on better algorithms instead of wasting resources on hyperparameter search
- Optimizer: 
  - Adam vs RMSprop: I see people often use RMSprop as A3C paper for A2C/A3C. Some people say RMSprop helps agents learn better but takes longer than Adam. Some people say Adam is definitely better. I tried it and found Adam is better for this project with my tests!
  - If you want to improve, spend time on better algorithms instead of wasting resources on hyperparameter search
  - Some people recomment RMSprop work better (especially tf version, torch and tf implement RMSprop have some small differents). You can see more at [A2C SB3](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html). This is their [tf like RMSprop implement](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/sb2_compat/rmsprop_tf_like.py)

| World | Stage | num_envs | learn_step | gamma | learning_rate | optimizer_eps | detach_lstm_state | init_weights | training_step | training_time   |
|-------|-------|----------|------------|-------|---------------|---------------|-----------------|---|---|---| 
| 1     | 1     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 374000        | 5:34:38         |
| 1     | 2     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 1388000       | 18:44:51        |
| 1     | 4     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 56000         | 0:51:34         |
| 2     | 1     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 2520000       | 15:11:00        |
| 2     | 2     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 2582000       | 1 day, 14:54:56 | 
| 2     | 3     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 401000        | 5:28:46         |
| 2     | 4     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 247000        | 3:40:49         |
| 3     | 1     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 308000        | 4:43:37         |
| 3     | 2     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 156000        | 2:29:00         |
| 3     | 3     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 1311000       | 15:16:28        |
| 3     | 4     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 443000        | 6:46:07         |
| 4     | 1     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 295000        | 4:08:22         |
| 4     | 2     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 1120000       | 11:35:43        |
| 5     | 1     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 486000        | 7:10:43         |
| 5     | 2     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 1089000       | 11:50:50        |
| 5     | 4     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 1636000       | 22:02:11        |
| 6     | 1     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 88000         | 1:01:50         |
| 6     | 2     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 1215000       | 16:41:19        |
| 6     | 3     | 16       | 20         | 0.99  | 7e-5          | 1e-8 | False | False | 1746500     | 13:50:27        |
| 6     | 4     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 940000        | 7:57:47         |
| 7     | 1     | 8        | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 528000        | 4:06:45         |
| 7     | 2     | 8        | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 3427000       | 1 day, 5:22:18  |
| 7     | 3     | 8        | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 1545000       | 12:17:16        |
| 7     | 4     | 16       | 20         | 0.99  | 1e-4          | 1e-8 | True | True | 1462000       | 15:27:19        |
| 8     | 1     | 16       | 5          | 0.9   | 1e-4          | 1e-8 | True | True | 2158000       | 1 day, 2:35:41  |
| 8     | 2     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 1107000       | 16:24:30        |
| 8     | 3     | 16       | 20         | 0.9   | 1e-4          | 1e-8 | True | True | 521000        | 8:12:27         |

### A3C

I don't have enough resources to run A3C (I also don't find A3C much better or worse than A2C). 

Here is the hyperparameter set I used to test the 4 states:

| World | Stage | num_envs | optimizer | optimizer_eps | gamma | learn_step | learning_rate | V_coef | entropy_coef | max_grad_norm | training_step | training_time   |
|-------|-------|----------|-----------|---------------|-------|------------|---------------|--------|--------------|--------------|---------------|-----------------|
| 1     | 1     | 16       | RMSprop   | 0.1           | 0.99  | 20         | 1e-4          | 0.5    | 0.01         | 40 | 119860        | 1:54:59         |
| 1     | 2     | 16       | RMSprop   | 0.1           | 0.99  | 20         | 1e-4          | 0.5    | 0.01         | 40 | 1492280       | 1 day, 17:24:07 |
| 1     | 4     | 16       | RMSprop   | 0.1           | 0.99  | 20         | 1e-4          | 0.5    | 0.01         | 40 | 32800         | 0:31:14         |
| 2     | 2     | 16       | RMSprop   | 0.1           | 0.99  | 20         | 1e-4          | 0.5    | 0.01         | 40 | 4611620       | 4 days, 3:11:24 |

I cannot complete the stages that A2C does not complete with this hyperparameter set. 

### ACKTR

**KFAC:**

- I tried changing the `learning_rate`, `max_learning_rate`, and `kl_clip` (in KFAC), but there was no significant improvement.
- I also tried adding `max_gradient_norm` (before `optimizer.step` or `KFAC.step`), but it didn’t work.
- The `value_fisher_coef` was set to 1 by default (like in Stable Baselines); I tried 0.5 but saw no improvement.
- `eigen_eps`: 
  - this parameter helps `torch.linalg.eigh` avoid one of the following three errors:
    - The algorithm failed to converge because the input matrix is ill-conditioned or has too many repeated eigenvalues
    - This error may appear if the input matrix contains NaN (i.e., the model contains NaNs)
    - Training stops without an obvious error (possibly because eigenvalues < 1e-6, leading to no gradient)
  - Initially, I set it to 1e-3. This value was quite large, and the agent couldn't learn the harder stages.
  - I reduced this parameter by a factor of 10 if the agent failed to learn. Stages 1-2, 3-3, 4-3, 5-3, 6-3 and 8-1 were completed with smaller `eigen_eps` values.
  - I also tried `eigen_eps = 1e-4` on some stages (2-1, 5-2, 7-4) and found that 1e-4 helped the agent learn significantly faster than 1e-3.

**ACKTR:**

- `entropy_coef = 0.05` did not help the agent learn better.
- `value_coef = 0.25` did not help the agent learn better.
- `gae_lambda = 0.95` performed better than 1.0 in one of my experiments.
- `gamma = 0.997` did not improve learning in my experiments.

**Hyperparameters:**

This is my ACKTR default hyperparameters:

| num_envs | learn_step | gamma | learning_rate | entropy_coef | V_coef | init_weights       | gae_lambda |
|----------|------------|-------|----------------|---------------|--------|---------------------|-------------|
| 32       | 20         | 0.99  | 5e-4           | 0.01          | 0.5    | True (orthogonal)   | 0.95        |

This is my KFAC default hyperparameters:

| momentum | stat_decay | Ts | Tf | kl_clip | damping | fast_cnn | weight_decay | learning_rate |
|----------|-------------|----|----|----------|----------|-----------|----------------|----------------|
| 0.9      | 0.99        | 1  | 10 | 0.001    | 1e-2     | False     | 0              | 5e-4           |

Training Results per Stage

| World | Stage | eigen_eps | Training Steps | Training Time       |
|-------|-------|-----------|----------------|---------------------|
| 1     | 1     | 1e-3      | 140500         | 1:45:35             |
| 1     | 2     | **1e-4**     | 223500         | 3:37:23             |
| 1     | 4     | 1e-3      | 46500          | 0:34:18             |
| 2     | 1     | 1e-3      | 4809500        | 2 days, 22:58:13    |
|       |       | 1e-4      | 884000         | 10:07:27            |
| 2     | 2     | 1e-3      | 704000         | 11:00:18            |
| 2     | 3     | 1e-3      | 144500         | 2:29:12             |
| 2     | 4     | 1e-3      | 134000         | 2:20:44             |
| 3     | 1     | 1e-3      | 125500         | 2:08:23             |
| 3     | 2     | 1e-3      | 80500          | 1:19:19             |
| 3     | 3     | **1e-4**      | 89500          | 1:30:21             |
| 3     | 4     | 1e-3      | 565500         | 9:12:46             |
| 4     | 1     | 1e-3      | 129500         | 2:16:04             |
| 4     | 2     | 1e-3      | 297000         | 5:46:47             |
| 4     | 3     | **1e-5**      | 148000         | 2:14:07             |
| 5     | 1     | 1e-3      | 120000         | 2:35:07             |
| 5     | 2     | 1e-3      | 796500         | 10:11:54            |
|       |       | 1e-4      | 433500         | 5:00:12             |
| 5     | 3     | **1e-6**      | 2551000        | 1 day, 19:54:33     |
| 5     | 4     | 1e-3      | 374500         | 7:00:12             |
| 6     | 1     | 1e-3      | 42500          | 0:51:21             |
| 6     | 2     | 1e-3      | 351000         | 7:01:03             |
| 6     | 3     | **1e-4**      | 443500         | 6:45:17             |
| 6     | 4     | 1e-3      | 64500          | 1:19:09             |
| 7     | 1     | 1e-3      | 360000         | 7:01:18             |
| 7     | 2     | 1e-3      | 1153000        | 17:14:12            |
| 7     | 3     | 1e-3      | 247500         | 4:08:02             |
| 7     | 4     | 1e-3      | 202000         | 2:38:51             |
|       |       | 1e-4      | 27000          | 0:16:03             |
| 8     | 1     | **1e-5**      | 1379500        | 22:58:53            |
|       |       | **1e-6**      | 428000         | 7:30:04             |
| 8     | 2     | 1e-3      | 525000         | 8:01:06             |
| 8     | 3     | 1e-3      | 478000         | 7:52:20             |

## Questions

* Is this code guaranteed to complete the stages if you try training?
  
This hyperparameter does not guarantee A2C will complete the stage, but I tried and most stages will complete on the first train. A few difficult stages will take 2 to 3 times to complete.

Stage 5-3 (ACKTR) will be very hard to reproduce (other stages will be guaranteed to reproduce in 1-3 random runs)

* How long do you train agents?
  
Within a few hours to more than 1 day. Time depends on hardware, I use many different hardware so time will not be accurate.

* How can you improve this code?
  
You can separate the test agent part into a separate thread or process (for A2C, ACKTR). I'm not good at multi-threaded programming so I don't do this.

* A3C leak memory?

  - Sometime, your machine will leak memory when run A3C (my code or other A3C codes), their are some reasons:
    - PyTorch or even Numpy version: try other Torch or Numpy versions.
    - Limit unnecessary variable creation and passing: maybe python can't clear memory efficiently than if we create or pass a lot of unnecessary variables, python can't clear it and yield leak memory.
    - Reuse logits and values when sample actions:
      - Because we only need train 1 time with each state. We can reuse logits and values when sample actions to backward.
      - I met leak memory problem when detach gradient in sample actions and forward states again in train function.
    
  - Adam take less memory than RMSprop. Than use Adam if you don't have enough memory for RMSprop.

  - Remember to observe the memory immediately when running A3C. If there is no memory leak, the memory will be occupied quickly and will not increase over time. If you see a slow increase, you have a memory leak (although memory leaks in A3C usually lead to memory overflow and are easy to detect)

* Compare A2C and A3C:
  - A3C is very difficult to implement and runs very slow compared to A2C. I also don't notice much difference between the two.
  - A3C uses a lot of memory. So i don't have enough resources to test enough for A3C.
  - Conclusion: should use A2C to save resources. If you want a better algorithm than A2C, use ACKTR, PPO or newer algorithms (Muzero, R2D2, Agent75, SAC, ...) instead of A3C (only marginally better in some environments and very resource-intensive, difficult to implement)

* Compare ACKTR and A2C:
  - Advantages:
    - ACKTR learns much faster than A2C.
    - ACKTR completes more stages (better than A2C)
    - ACKTR can use larger num_envs, learn_step (according to ACKTR paper, A2C will drop performance if batch_size=learn_step * num_envs is too large (~640) while ACKTR will be better if batchsize is increased).
    - Adding KFAC is very easy, just recode (or copy) the KFAC optimizer. Add a few lines of backward fisher matrix code

  - Disadvantages:
    - ACKTR cannot take advantage of LSTM, if KFAC can still work with LSTM, ACKTR's performance can be significantly better.
    - ACKTR has bugs if eigen_eps is not added! However, adding eigen_eps will distort the gradient (all torch KFAC implementations that I saw will have bugs so I added eigen_eps)
    - ACKTR cannot be as good as PPO even though KFAC is much more complex clipping in PPO
    - If you recode KFAC (instead of copying like me), it will be very difficult to implement because KFAC is complex and difficult to understand.
  
  - Conclusion: ACKTR is definitely better than A2C

## Requirements

* **python 3>3.6**
* **gym==0.25.2**
* **gym-super-mario-bros==7.4.0**
* **imageio**
* **imageio-ffmpeg**
* **cv2**
* **pytorch** 
* **numpy==1.26.4**

Note: newer version of numpy yield bug with Mario environment!

## Acknowledgements
With my A2C code, I can completed 27/32 stages of Super Mario Bros. With harder stages like 1-3, 5-3, ... A2C can not completed this stages.

With my ACKTR code, I can completed 29/32 stages of Super Mario Bros. With harder stages 1-3, 4-4, 8-4, ACKTR can not completed this stages.

## Reference
* [ikostrikov ACKTR](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
* [stable_baselines ACKTR](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/acktr/kfac.py)
* [uvipen A3C](https://github.com/uvipen/Super-mario-bros-A3C-pytorch)
* [uvipen PPO](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)
* [lazyprogrammer A2C](https://github.com/lazyprogrammer/machine_learning_examples/tree/master/rl3/a2c)
* [gianluca-maselli A3C](https://github.com/gianluca-maselli/A3C)
* [A2C SB3](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
* [A3C paper](https://arxiv.org/pdf/1602.01783)
* [ACKTR paper](https://arxiv.org/pdf/1708.05144)