强化学习简介
===============================================================

强化学习（Reinforcement Learning）是一类非常重要的机器学习方法。其目标是通过与环境的交互反馈，使智能体（agent）学习到激励最大化的行动策略（behave）。这些年，我们见证了强化学习取得了令人难以置信的进展。其中比较突出的，包括AlphaGo（DeepMind and the Deep Q learning architecture in 2014），OpenAI和PPO（2017）。

在这个系列中，我们将重点关注解决强化学习问题的不同方法与架构，包括：Q-learning，Deep Q-learning，Policy Gradient，Actor Critic和PPO。

在本章节将介绍：

- 强化学习的基本思想：什么是强化学习？以及激励反馈是如何成为强化学习的核心思想的
- 强化学习的3种重要方法
- 深度强化学习中的『深度』是什么意思？

掌握这些对我们深入理解并实现深度强化学习非常重要。

.. admonition:: 前置知识: Markov Decision Process

	在概率论和统计学中，*Markov \ Decision \ Processes* (MDP) 提供了一个数学架构模型，刻画的是"如何在部分随机，部分可由决策者控制的状态下进行决策"的过程。强化学习的体系正是构建在MDP之上的。

	一个 *Markov* *Decision* *Process* 是由这样元组 :math:`\left \langle S, A, P, R, \gamma \right \rangle` 组成，其中:  

    	* :math:`\mathcal{S}` 是一系列状态的集合；
    	* :math:`\mathcal{A}` 是一系列动作的集合；
    	* :math:`\mathcal{P}` 是状态转义概率矩阵，表示从状态 :math:`s` 采取行动策略 :math:`a` ，迁移到状态 :math:`s'` 的概率： :math:`P_{ss'}^a = \mathbb{P}[S_{t+1}=s' | S_t=s, A_t=a]`
    	* :math:`\mathcal{R}` 是激励函数，:math:`R_s^a=\mathbb{E}[R_{t+1}|S_t=s, A_t=a]`
    	* :math:`\mathcal{\gamma}` 是折扣因子，:math:`\gamma \in [0,1]`

	在这个基础之上，自然引申出 :math:`\mathbf{policy}` 和 :math:`\mathbf{return}` 的概念。

    	* Policy  :math:`\pi`   是状态 :math:`s` 和行动策略 :math:`a` 的分布函数: :math:`\pi(a|s) = \mathbb{P}[A_{t}=a | S_t=s]`
    	* Return  :math:`G_t`  是从t时刻开始的累计期望激励 :math:`G_t = R_{t+1} + R_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^{k}R_{t+k+1}`
	
	:math:`\mathbf{value \ function}` 也是MDP中一个非常重要的概念，衡量的是从某个状态开始计算的return期望值，value function一般有两种定义方式:

    	* :math:`\mathbf{state\!-\!value \ function}` : \ :math:`v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]` ，表示的是从状态 :math:`s` ，遵循Policy :math:`\pi` 的期望return
    	* :math:`\mathbf{action\!-\!value \ function}` : \ :math:`q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]` ，则还考虑到了在时刻t的动作

	从定义上看， :math:`v_{\pi}(s)` 和 :math:`q_{\pi}` 是可以相互转换的:

    	* :math:`v_{\pi}(s) = \sum_{a \in A} \pi(a|s) q_{\pi}(s, a)` 
    	* :math:`q_{\pi}(s,a) = \mathcal{R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s')}` 



强化学习的基本思想
^^^^^^^^^^^^^^^^^^^^^^^^^^^

强化学习的核心思想，是通过与环境的交互、以及环境给予的反馈中，来学习使激励最大化的行为方法。通过与环境的交互反馈来进行学习是人的自然技能。假设你是一个小孩，在客厅里看到了一个壁炉。

	.. figure:: ./images/fire1.png
		:width: 50%
		:align: center

你靠近它，这时你感觉到很温暖（正反馈+1），你学习到"火"可以给你带来正反馈。

	.. figure:: ./images/fire2.png
		:width: 60%
		:align: center

但是当你想要去触碰火焰时，火焰会灼伤你的手，这时你会收到一个负反馈，你会感觉到痛（负反馈-1）。通过交互与激励反馈，你学习到的是：火焰在一定范围的距离外，可以提供温暖（正反馈）；但是距离太近，就会被烧伤（负反馈）。这就是人，如何通过与环境的交互进行学习的过程，强化学习主要是利用了这个思想，在Action->reward的过程中，学习使整体reward最大的策略。

	.. figure:: ./images/fire3.png
		:width: 60%
		:align: center


强化学习的过程
^^^^^^^^^^^^^^^^^^^^^^^^^^^

以超级玛丽游戏为例，通过强化学习训练一个能玩游戏的智能体（Agent），其过程可以按如下环节进行建模：

- 智能体从环境（Env.）获取的初始状态（State）为S0。在当前示例中，我们取游戏的第一帧（first frame）作为S0。
- 基于状态S0，智能体采用行动（Action）策略A0。在当前示例中，智能体将向右走一步。
- 当前环境的状态迁移到S1（新的一帧）。
- 环境根据当前的状态（人物还没死亡，reward+1），给智能体激励（reward）反馈R1。

	.. figure:: ./images/superMario.png
		:width: 100%
		:align: center

强化学习的过程就是不断循环上述过程，并输出（state，action，reward）序列。智能体的目标就是使期望累计reward最大化。