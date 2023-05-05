import numpy as np
from deepbots.supervisor.controllers.supervisor_emitter_receiver import SupervisorCSV
from PPOAgent import PPOAgent, Transition
from agent import NAF_Agent
from utilities import normalizeToRange
from scipy.spatial import distance
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import gym
from gym import spaces
from sklearn import preprocessing
    
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
		super().__init__()
		self.observationSpace = 27  # Define the number of inputs for the agent (27)
		h = np.ones(self.observationSpace)*100 #Define lower and upper bounds of observations -- values between -100 and 100 not accurate
		l = np.ones(self.observationSpace)*100
		self.observation_space = spaces.Box(np.array(h*-1), np.array(h)) 
		self.actionSpace = 7  # The agent can perform 7 actions, one velocity per motor
		self.action_space = spaces.Box(-np.ones(self.actionSpace), np.ones(self.actionSpace))
		self.robot = None
		self.respawnRobot()
		self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT") #Used top create a simple pole obstacle
		self.messageReceived = None	 # Variable to save the messages received from the robot
		
		self.episodeCount = 0  # Episode counter
		self.episodeLimit = 10000  # Max number of episodes allowed
		self.stepsPerEpisode = 2000  # Max number of steps per episode
		self.episodeScore = 0  # Score accumulated during an episode
		self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
		self.action = 0

	def respawnRobot(self):
		if self.robot is not None:
			# Despawn existing robot
			self.robot.remove()

		# Respawn robot in starting position and state
		rootNode = self.supervisor.getRoot()  # This gets the root of the scene tree
		childrenField = rootNode.getField('children')  # This gets a list of all the children, ie. objects of the scene
		childrenField.importMFNode(-2, "robot.wbo")	 # Load robot from file and add to second-to-last position

		# Get the new robot and pole endpoint references
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
        obsR = (dref/ (dobs + dref))**p
        rew = -d - np.linalg.norm(self.action) + obsR

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
	
	def learn(self, agent):
		
		param_noise = None
		action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(supervisor.actionSpace), sigma=float(0.5) * np.ones(supervisor.actionSpace))

		solved = False
		# Run outer loop until the episodes limit is reached or the task is solved
		while not solved and self.episodeCount < self.episodeLimit:
			observation = self.reset()  # Reset robot and get starting observation
			self.episodeScore = 0
			
			
			for step in range(self.stepsPerEpisode):
				# In training mode the agent samples from the probability distribution, naturally implementing exploration
				print(np.asarray(observation).flatten)
				selectedAction = agent.act(np.asarray(observation).flatten())
				# Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
				# the done condition
				self.action = selectedAction
				newObservation, reward, done, info = self.step(selectedAction)

				# Save the current state transition in agent's memory
				trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
				agent.storeTransition(trans)
				
				if done:
					# Save the episode's score
					self.episodeScoreList.append(self.episodeScore)
					agent.step(observation, selectedAction, reward, newObservation, done)
					solved = self.solved()  # Check whether the task is solved
					break

				self.episodeScore += reward  # Accumulate episode reward
				observation = newObservation  # observation for next step is current step's newObservation
			print("Episode #", self.episodeCount, "score:", self.episodeScore)
			self.episodeCount += 1  # Increment episode counter

		if not solved:
			print("Task is not solved, deploying agent for testing...")
		elif solved:
			print("Task is solved, deploying agent for testing...")
			
		observation = self.reset()
		while True:
			selectedAction = agent.act(np.asarray(observation))
			observation, _, _, _ = self.step(selectedAction)

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-info", type=str, default="Experiment-1",
						help="Name of the Experiment (default: Experiment-1)")
	parser.add_argument('-env', type=str, default="Manipulator-v0",
						help='Name of the environment (default: Pendulum-v0)')
	parser.add_argument('-f', "--frames", type=int, default=40000,
						help='Number of training frames (default: 40000)')   
	parser.add_argument("--eval_every", type=int, default=5000,
						help="Evaluate the current policy every X steps (default: 5000)")
	parser.add_argument("--eval_runs", type=int, default=2,
						help="Number of evaluation runs to evaluate - averating the evaluation Performance over all runs (default: 3)")
	parser.add_argument('-mem', type=int, default=100000,
						help='Replay buffer size (default: 100000)')
	parser.add_argument('-per', type=int, choices=[0,1],  default=0,
						help='Use prioritized experience replay (default: False)')
	parser.add_argument('-b', "--batch_size", type=int, default=128,
						help='Batch size (default: 128)')
	parser.add_argument('-nstep', type=int, default=1,
						help='nstep_bootstrapping (default: 1)')
	parser.add_argument("-d2rl", type=int, choices=[0,1], default=1,
						help="Using D2RL Deep Dense NN Architecture if set to 1 (default: 0)")
	parser.add_argument('-l', "--layer_size", type=int, default=256,
						help='Neural Network layer size (default: 256)')
	parser.add_argument('-g', "--gamma", type=float, default=0.99,
						help='Discount factor gamma (default: 0.99)')
	parser.add_argument('-t', "--tau", type=float, default=0.005,
						help='Soft update factor tau (default: 0.005)')
	parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
						help='Learning rate (default: 1e-3)')
	parser.add_argument('-u', "--update_every", type=int, default=1,
						help='update the network every x step (default: 1)')
	parser.add_argument('-n_up', "--n_updates", type=int, default=1,
						help='update the network for x steps (default: 1)')
	parser.add_argument('-s', "--seed", type=int, default=0,
						help='random seed (default: 0)')
	parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip gradients (default: 1.0)")
	parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse", help="Choose loss type MSE or Huber loss (default: mse)")

	args = parser.parse_args()
		#wandb.init(project="naf", name=args.info)
		#wandb.config.update(args)
	writer = SummaryWriter("runs/"+args.info)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print("Using ", device)
		
	env = PepperSupervisor()
	agent = NAF_Agent(state_size=self.observationSpace,
						action_size=self.actionSpace,
						device=device, 
						args=args,
						writer=writer)
	env.learn(agent)