
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from scipy.spatial import distance
from gym import spaces
import os
import time
from collections import deque
import pickle

from baselines.ddpg.ddpg_learner import DDPG
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U

from baselines import logger
import numpy as np


class PepperSupervisor(SupervisorCSV):
    """Pepper supervisor environment, acts as Open AI gym environment.

        Attributes:
            self.observationSpace: Number of observations, Int.
            self.observation_space: Gym space object describing the observation space.
            self.action_space: Gym space object describing the action space.
            self.actionSpace: Number of actions, Int.
            self.robot: Webots robot object.
            self.target: Webots target object.
            self.endPoint: Webots endpoint of robot object.
            self.messageReceived: Variable to save the messages received from the robot controller.
            self.episodeCount: Episode counter, Int.
            self.episodeLimit:  Max number of episodes allowed, Int.
            self.stepsPerEpisode: Max number of steps per episodem Int
            self.episodeScore: Score accumulated during an episode, Float.
            self.episodeScoreList: List containing all the past episode scores.
            self.action: Current proposed action from the the agent.
            self.joint1: Webots solid object describing joint 1.
            self.joint2: Webots solid object describing joint 2.
            self.joint3: Webots solid object describing joint 3.
            self.joint4: Webots solid object describing joint 4.
            self.joint5: Webots solid object describing joint 5.
            self.joint6: Webots solid object describing joint 6.
            self.joint7: Webots solid object describing joint 7.
        """
    def __init__(self):
        """Inits the pepper supervisors environment.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.observationSpace = 27  # The agent can observe x actions, used in replacement of action_space depending on the agent
        l = [-10,-10,-10,-10,-10,-10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        h = [10,10,10,10, 10,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        self.observation_space = spaces.Box(np.array(l), np.array(h))  # Generate observation space based on low and high limits
        self.actionSpace = 7  # The agent can perform 7 actions, used in replacement of action_space depending on the agent
        self.action_space = spaces.Box(np.array([-1,-1,-1,-1,-1,-1,-1]), np.array([+1,+1,+1,+1,+1,+1,+1])) # Define actions space between -1 and 1 for each joint (DDPG output always between -1 and 1, scale for desired output)
        self.robot = None
        self.respawnRobot() # Initially spawn the robot
        self.messageReceived = None	 # Variable to save the messages received from the robot controller

        self.episodeCount = 0  # Episode counter
        self.episodeLimit = 10000  # Max number of episodes allowed
        self.stepsPerEpisode = 2000  # Max number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        self.action = 0
    def respawnRobot(self):
        """Inits SampleClass with blah.

        Args:
            None

        Returns:
            None
        """
        if self.robot is not None:
            # Despawn existing robot
            self.robot.remove()

        # Respawn robot in starting position and state
        rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
        childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
        childrenField.importMFNode(-2, "robot.wbo")	 # Load robot from file and add to second-to-last position

        # Get the new robot and target references
        self.robot = self.supervisor.getFromDef("ROBOT")
        self.target = self.supervisor.getFromDef("TARGET")
        self.endpoint = self.supervisor.getFromDef("END_POINT")
        self.obstacle = self.supervisor.getFromDef("OBSTACLE")
        self.joint1 = self.supervisor.getFromDef("Joint_1")
        self.joint2 = self.supervisor.getFromDef("Joint_2")
        self.joint3 = self.supervisor.getFromDef("Joint_3")
        self.joint4 = self.supervisor.getFromDef("Joint_4")
        self.joint5 = self.supervisor.getFromDef("Joint_5")
        self.joint6 = self.supervisor.getFromDef("Joint_6")
        self.joint7 = self.supervisor.getFromDef("Joint_7")


    def get_observations(self):
        """Inits SampleClass with blah.

        Args:
            None

        Returns:
            A 1-D numpy array containing the observations perceived in the Webots world.
        """
        # Position on z axis, third (2) element of the getPosition vector
        endPosition = self.endpoint.getPosition()
        targetPosition = self.target.getPosition()
        obstaclePosition = self.obstacle.getPosition()
        dis = [endPosition[0] - targetPosition[0], endPosition[1] - targetPosition[1], endPosition[2] - targetPosition[2]]
        # Update self.messageReceived received from robot, which contains pole angle
        self.messageReceived = self.handle_receiver()
        # jointPos = [self.joint1.getPosition(), self.joint2.getPosition(), self.joint3.getPosition(), self.joint4.getPosition(), self.joint5.getPosition(), self.joint6.getPosition(), self.joint7.getPosition()]
        
        if self.messageReceived is not None:
             robotVals = [float(i) for i in self.messageReceived]
        else:
        # Method is called before self.messageReceived is initialized
            robotVals = [0.0 for i in range(21)]
        obs = np.array([targetPosition + dis + robotVals])
        return obs

    def get_reward(self, action):
        """Calculates the current steps reward observed given the performed action chosen by the agent.

                Args:
                    action: Action performed on the environment.

                Returns:
                    A float describing the current steps reward based on the performed action.
        """
        obstaclePosition = self.obstacle.getPosition()
        targetPosition = self.target.getPosition()
        endPosition = self.endpoint.getPosition()
        d = distance.euclidean(targetPosition, endPosition)
        dobs = distance.euclidean(obstaclePosition, endPosition)
        dref = 5  # constant parameter so that 0 < obsR < 1
        p = 1  # p is for the exponential decay of negative reward
        # delt = 0.4
        # if d < delt:
        # rew = 0.5 * d**2
        # else:
        # rew = delt * (abs(d) - 0.5 * delt)
        obsR = (dref/ (dobs + dref))**p
        rew = -d - np.linalg.norm(self.action)

        return rew

    def is_done(self):
        """Checks to see if the goal of the robot was achieved.

                Args:
                    None

                Returns:
                    A list containing a boolean describing whether the finished criteria was met.
        """
        targetPosition = self.target.getPosition()
        endPosition = self.endpoint.getPosition()

        d = distance.euclidean(targetPosition, endPosition)
        if d <= 0.1:
            return [True]

        return [False]

    def solved(self):
        """Checks if the RL training has reached a satisfactory level of success.

                Args:
                    None

                Returns:
                    A boolean describing if the RL training has 'solved' the problem.
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False

    def reset(self):
        """Resets the simulation environment.

                Args:
                    None

                Returns:
                    A numpy array describing the observation space, all with zero observation.
        """
        self.respawnRobot()
        self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
        self.messageReceived = None
        return np.asarray([[0.0 for _ in range(self.observationSpace)]])

    def get_info(self):
        """Grabs information describing the environment.

                Args:
                    None

                Returns:
                    None
        """
        return None

def learn(network, env,
          seed=None,
          total_timesteps=None,
          nb_epochs=None, # with default settings, perform 1M steps total
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50, # per epoch cycle and MPI worker,
          nb_eval_steps=100,
          batch_size=64, # per MPI worker
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs):
    """Resets the simulation environment.

            Args:
                None

            Returns:
                network: The network to be used, (MLP, etc.)
                env: The environment to learn from, Gym Environment
                seed: Seed for the pseudo-random generators (python, numpy, tensorflow). If None (default), use random seed, int.
                total_timesteps: The total number of samples to train on, int.
                nb_epochs: The number of epochs during training, int.
                nb_epoch_cycles: The number of epoch cycles during training, int.
                nb_rollout_steps: The number of rollout steps, int.
                reward_scale: The value the reward should be scaled by, float.
                render: Enable rendering of the environment, bool.
                render_eval: Enable rendering of the evaluation environment, bool
                noise_type: Noise type, can be none.
                normalize_returns: should the returns be normalized, bool
                normalize_observations: Should the observation be normalized, bool.
                critic_l2_reg: l2 regularizer coefficient, float.
                actor_lr: The actor learning rate, float.
                critic_lr: The critic learning rate, float.
                popart: Enable pop-art normalization of the critic output (https://arxiv.org/pdf/1602.07714.pdf), normalize_returns must be set to True, bool.
                gamma: The discount factor, float.
                clip_norm: Clip the gradients (disabled if None), float.
                nb_train_steps: The number of training steps, int.
                nb_eval_steps: The number of evaluation steps, int.
                batch_size: The size of the batch for learning the policy, int.
                tau: The soft update coefficient (keep old values, between 0 and 1), float.
                eval_env: The evaluation environment, Gym Environment.
                param_noise_adaption_interval: Apply param noise every N steps, int.
                **network_kwargs: Additional arguments to be passed to the network on creation.
    """

    set_global_seeds(seed)

    if total_timesteps is not None:
        assert nb_epochs is None
        nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
    else:
        nb_epochs = 500

    rank = 0

    nb_actions = env.action_space.shape[-1]
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.

    # Initialize the memory, critic, and actor objects used in DDPG
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(network=network, **network_kwargs)
    actor = Actor(nb_actions, network=network, **network_kwargs)

    # Add noise to the signal, using built in noise objects
    action_noise = None
    param_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    max_action = env.action_space.high
    logger.info('scaling actions by {} before executing in env'.format(max_action))

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    logger.info('Using agent with the following configuration:')
    logger.info(str(agent.__dict__.items()))

    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)
    sess = U.get_session()

    # Prepare everything.
    agent.initialize(sess)
    sess.graph.finalize()

    agent.reset()

    obs = env.reset()
    if eval_env is not None:
        eval_obs = eval_env.reset()
    nenvs = obs.shape[0]

    episode_reward = np.zeros(nenvs, dtype = np.float32) #vector
    episode_step = np.zeros(nenvs, dtype = int) # vector
    episodes = 0 #scalar
    t = 0 # scalar

    epoch = 0

    start_time = time.time()

    epoch_episode_rewards = []
    epoch_episode_steps = []
    epoch_actions = []
    epoch_qs = []
    epoch_episodes = 0
    for epoch in range(nb_epochs):
        for cycle in range(nb_epoch_cycles):
            # Perform rollouts.\
            obs = env.reset()
            agent.reset()
            if nenvs > 1:
                # if simulating multiple envs in parallel, impossible to reset agent at the end of the episode in each
                # of the environments, so resetting here instead
                agent.reset()
            for t_rollout in range(nb_rollout_steps):
                # Predict next action.
                action, q, _, _ = agent.step(obs, apply_noise=True, compute_Q=True)

                # Execute next action.
                if rank == 0 and render:
                    env.render()

                # max_action is of dimension A, whereas action is dimension (nenvs, A) - the multiplication gets broadcasted to the batch
                new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                # note these outputs are batched from vecenv

                t += 1
                if rank == 0 and render:
                    env.render()
                episode_reward += r
                episode_step += 1

                # Book-keeping.
                epoch_actions.append(action)
                epoch_qs.append(q)
                agent.store_transition(obs, action, r, new_obs, done) #the batched data will be unrolled in memory.py's append.

                obs = new_obs

                for d in range(len(done)):
                    if done[d]:
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward[d])
                        episode_rewards_history.append(episode_reward[d])
                        epoch_episode_steps.append(episode_step[d])
                        episode_reward[d] = 0.
                        episode_step[d] = 0
                        epoch_episodes += 1
                        episodes += 1
                        if nenvs == 1:
                            agent.reset()



            # Train.
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []
            for t_train in range(nb_train_steps):
                # Adapt param noise, if necessary.
                if memory.nb_entries >= batch_size and t_train % param_noise_adaption_interval == 0:
                    distance = agent.adapt_param_noise()
                    epoch_adaptive_distances.append(distance)

                cl, al = agent.train()
                epoch_critic_losses.append(cl)
                epoch_actor_losses.append(al)
                agent.update_target_net()

            # Evaluate.
            eval_episode_rewards = []
            eval_qs = []
            if eval_env is not None:
                nenvs_eval = eval_obs.shape[0]
                eval_episode_reward = np.zeros(nenvs_eval, dtype = np.float32)
                for t_rollout in range(nb_eval_steps):
                    eval_action, eval_q, _, _ = agent.step(eval_obs, apply_noise=False, compute_Q=True)
                    eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    if render_eval:
                        eval_env.render()
                    eval_episode_reward += eval_r

                    eval_qs.append(eval_q)
                    for d in range(len(eval_done)):
                        if eval_done[d]:
                            eval_episode_rewards.append(eval_episode_reward[d])
                            eval_episode_rewards_history.append(eval_episode_reward[d])
                            eval_episode_reward[d] = 0.0


        mpi_size = 1

        # Log stats.
        # XXX shouldn't call np.mean on variable length lists
        duration = time.time() - start_time
        stats = agent.get_stats()
        combined_stats = stats.copy()
        combined_stats['rollout/return'] = np.mean(epoch_episode_rewards)
        combined_stats['rollout/return_std'] = np.std(epoch_episode_rewards)
        combined_stats['rollout/return_history'] = np.mean(episode_rewards_history)
        combined_stats['rollout/return_history_std'] = np.std(episode_rewards_history)
        combined_stats['rollout/episode_steps'] = np.mean(epoch_episode_steps)
        combined_stats['rollout/actions_mean'] = np.mean(epoch_actions)
        combined_stats['rollout/Q_mean'] = np.mean(epoch_qs)
        combined_stats['train/loss_actor'] = np.mean(epoch_actor_losses)
        combined_stats['train/loss_critic'] = np.mean(epoch_critic_losses)
        combined_stats['train/param_noise_distance'] = np.mean(epoch_adaptive_distances)
        combined_stats['total/duration'] = duration
        combined_stats['total/steps_per_second'] = float(t) / float(duration)
        combined_stats['total/episodes'] = episodes
        combined_stats['rollout/episodes'] = epoch_episodes
        combined_stats['rollout/actions_std'] = np.std(epoch_actions)
        # Evaluation statistics.
        if eval_env is not None:
            combined_stats['eval/return'] = eval_episode_rewards
            combined_stats['eval/return_history'] = np.mean(eval_episode_rewards_history)
            combined_stats['eval/Q'] = eval_qs
            combined_stats['eval/episodes'] = len(eval_episode_rewards)
        def as_scalar(x):
            if isinstance(x, np.ndarray):
                assert x.size == 1
                return x[0]
            elif np.isscalar(x):
                return x
            else:
                raise ValueError('expected scalar, got %s'%x)

        combined_stats_sums = np.array([ np.array(x).flatten()[0] for x in combined_stats.values()])
        if MPI is not None:
            combined_stats_sums = MPI.COMM_WORLD.allreduce(combined_stats_sums)

        combined_stats = {k : v / mpi_size for (k,v) in zip(combined_stats.keys(), combined_stats_sums)}

        # Total statistics.
        combined_stats['total/epochs'] = epoch + 1
        combined_stats['total/steps'] = t

        for key in sorted(combined_stats.keys()):
            logger.record_tabular(key, combined_stats[key])

        if rank == 0:
            logger.dump_tabular()
        logger.info('')
        logdir = logger.get_dir()
        if rank == 0 and logdir:
            if hasattr(env, 'get_state'):
                with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                    pickle.dump(env.get_state(), f)
            if eval_env and hasattr(eval_env, 'get_state'):
                with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                    pickle.dump(eval_env.get_state(), f)


    return agent


if __name__ == "__main__":
    MPI = None
    env = PepperSupervisor()
    env.get_observations()
    learn('mlp', env, batch_size=128, nb_rollout_steps=2000)
