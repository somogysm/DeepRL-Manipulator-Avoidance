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
#from stable_baselines.ddpg.policies import MlpPolicy
#from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
#from stable_baselines import DDPG

    
class PepperSupervisor(SupervisorCSV):
	def __init__(self):
		super().__init__()
		self.observationSpace = 27  # The agent has 4 inputs
		l = [-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100]
		h = [100,100,100,100, 100,100,100,100,100,100,100,100,100]
		self.observation_space = spaces.Box(np.array(l), np.array(h))  # The agent has 4 inputs
		self.actionSpace = 7  # The agent can perform 2 actions
		#self.action_space = spaces.Discrete(4)  # The agent can perform 2 actions
		self.action_space = spaces.Box(np.array([-1,-1,-1,-1,-1,-1,-1]), np.array([+1,+1,+1,+1,+1,+1,+1]))
		self.robot = None
		self.respawnRobot()
		self.poleEndpoint = self.supervisor.getFromDef("POLE_ENDPOINT")
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

	def get_observations(self):
		# Position on z axis, third (2) element of the getPosition vector
		endPosition = self.endpoint.getPosition()
		#endVel = self.endpoint.getVelocity()
		# Linear velocity on z axis
		#cartVelocity = normalizeToRange(self.robot.getVelocity()[2], -0.2, 0.2, -1.0, 1.0, clip=True)
		# Angular velocity x of endpoint
		targetPosition = self.target.getPosition()
		dis = [endPosition[0] - targetPosition[0], endPosition[1] - targetPosition[1], endPosition[2] - targetPosition[2]] 
		# Update self.messageReceived received from robot, which contains pole angle
		self.messageReceived = self.handle_receiver()
		if self.messageReceived is not None:
			robotVals = [float(i) for i in self.messageReceived]
		else:
			# Method is called before self.messageReceived is initialized
			robotVals = [0.0 for i in range(21)]
        	
	
		return preprocessing.normalize(np.array([targetPosition + dis + robotVals]))
		
	def get_reward(self, action):
		targetPosition = self.target.getPosition()
		endPosition = self.endpoint.getPosition()
		d = distance.euclidean(targetPosition, endPosition)
		# delt = 0.4
		# if d < delt:
			# rew = 0.5 * d**2
		# else:
			# rew = delt * (abs(d) - 0.5 * delt)
			
		rew = -d - np.linalg.norm(self.action)

		return rew
	
	def is_done(self):
		if self.messageReceived is not None:
			robotVals = self.messageReceived
			for i in range(7):
            			vars()['Joint ' + str(i+1) +' position'] = list(robotVals[i])[0]
            			vars()['Joint ' + str(i+1) +' velocity'] = list(robotVals[i])[1]
                    		
		else:
			# method is called before self.messageReceived is initialized
			for i in range(7):
            			vars()['Joint ' + str(i+1) +' position'] = 0
            			vars()['Joint ' + str(i+1) +' velocity'] = 0
                    		
                            		

		# if self.episodeScore > 250.0:
			# return True

		targetPosition = self.target.getPosition()
		endPosition = self.endpoint.getPosition()
		
		d = distance.euclidean(targetPosition, endPosition)
		if d <= 0.1:
			return True

		return False
	
	def solved(self):
		if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
			if np.mean(self.episodeScoreList[-100:]) > 195.0:  # Last 100 episodes' scores average value
				return True
		return False
		
	def reset(self):
		self.respawnRobot()
		self.supervisor.simulationResetPhysics()  # Reset the simulation physics to start over
		self.messageReceived = None
		return [0.0 for _ in range(self.observationSpace)]
		
	def get_info(self):
		return None
		
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
parser.add_argument("-d2rl", type=int, choices=[0,1], default=0,
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
    
supervisor = PepperSupervisor()
agent = NAF_Agent(state_size=supervisor.observationSpace,
                      action_size=supervisor.actionSpace,
                      device=device, 
                      args= args,
                      writer=writer)
param_noise = None
action_noise = None#OrnsteinUhlenbeckActionNoise(mean=np.zeros(supervisor.actionSpace), sigma=float(0.5) * np.ones(supervisor.actionSpace))

#model = DDPG(MlpPolicy, supervisor, verbose=1, param_noise=param_noise, action_noise=action_noise)
solved = False
# Run outer loop until the episodes limit is reached or the task is solved
while not solved and supervisor.episodeCount < supervisor.episodeLimit:
	observation = supervisor.reset()  # Reset robot and get starting observation
	supervisor.episodeScore = 0
	
	
	for step in range(supervisor.stepsPerEpisode):
		# In training mode the agent samples from the probability distribution, naturally implementing exploration
		print(np.asarray(observation).flatten)
		selectedAction= agent.act(np.asarray(observation).flatten())
		#selectedAction, _states = model.predict(observation)
		# Step the supervisor to get the current selectedAction's reward, the new observation and whether we reached 
		# the done condition
		supervisor.action = selectedAction
		newObservation, reward, done, info = supervisor.step(selectedAction)

		# Save the current state transition in agent's memory
		#trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
		#agent.storeTransition(trans)
		
		if done:
			# Save the episode's score
			supervisor.episodeScoreList.append(supervisor.episodeScore)
			agent.step(observation, selectedAction, reward, newObservation, done)
			solved = supervisor.solved()  # Check whether the task is solved
			break

		supervisor.episodeScore += reward  # Accumulate episode reward
		observation = newObservation  # observation for next step is current step's newObservation
	print("Episode #", supervisor.episodeCount, "score:", supervisor.episodeScore)
	supervisor.episodeCount += 1  # Increment episode counter

if not solved:
	print("Task is not solved, deploying agent for testing...")
elif solved:
	print("Task is solved, deploying agent for testing...")
	
observation = supervisor.reset()
while True:
	selectedAction = agent.act(np.asarray(observation))
	#selectedAction, _states = model.predict(observation)
	observation, _, _, _ = supervisor.step(selectedAction)
