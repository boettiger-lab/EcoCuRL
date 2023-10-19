# EcoCuRL

*Curriculum learning tools for ecological decision problems under model uncertainty.*

Mathematical models for ecological systems are often used to guide decision-making, particularly so when there are competing goals that are to be achieved.
A classic example of this is fishery management, where the goal of profit is balanced with the goal of sustainability.
Similarly, models play a role in problems associated to allocating resources for environmental goals---for example invasive species management, or conservation of endangered species.

Using models to find optimal decision-making strategies, though, is tricky since the models used are often minimalistic and have biases. 
For example, the maximum sustainable yield (MSY) solution to the fishery management problem, provably optimal for certain models, can lead to overfishing in practice.
This requirement of simplicity in the model is endemic to optimal control theory.

Recently, *reinforcement learning* (RL) has been proposed as a technique which could alleviate this by allowing more complex -- more precise -- models to be used.
In [a recent publication](https://arxiv.org/abs/2308.13654), we showed that in certain regimes, RL based techniques can outperform classic solutions like MSY and constant escapement.

One outstanding question not addressed in that work is: even if RL allows for more expressive and complex models to be used, can how do we deal with the residual model uncertainty?
This work aims to approach this question from the reinforcement learning perspective, in particular using tools from *curriculum learning*.
