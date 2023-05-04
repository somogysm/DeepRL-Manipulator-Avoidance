# Reinforcment Learning using Webots simulation

This repository uses Open AI Gym along with Webots phyisics simulation to control a 7 DOF Kinova manipulator robot path movement towards a desired goal with obstacle avoidance, using reinforcment learning.

![](RL.gif)

## Prerequisites


### Install Webots!

- Install Webots [HERE](https://cyberbotics.com/#download)


### Install Required Python Packages!

pip install -r requirements.txt


## USAGE
-From within Webots, navigate to the File tab, and select open world 

-Browse to pepper_dss\algorithms\RL_Algorithms\pepper_webots and open the kinova_gen3_mode_Test.wbt

-Run simply by pressing the play button at the top of the webots UI. (Depending on hardware, the software may freeze. To avoid this quickly initiate the simulation by pressing play, and again pressing it to pause. This will begin loading in the 3D model of the robot into the simulation, once the robot is finished loading you may then continue the simulation by pressing play again. From here on out the software will run fine, upon finishing an episode the robot is destroyed and reloaded into the world at the initial point, this causes the software to temporarily freeze. Simply wait for the robot to load in again, and the simulation will continue automatically.

-The robot model is saved as Robot.wbo, located at pepper_dss\algorithms\RL_Algorithms\pepper_webots\controllers\supervisorController.

-The robot is controlled using the webots robot controller architecture. The python script doing so is located at pepper_dss\algorithms\RL_Algorithms\pepper_webots\controllers\Kinova\Kinova.py

-The robot contoller communicated with the supervisor controller located at pepper_dss\algorithms\RL_Algorithms\pepper_webots\controllers\supervisorController\supervisorController.py. The supervisor controller effectively acts as the Open AI Gym environment, this directly interacts with the RL Agents being used. Decisisions performed by the agent are then communicated back to the robot controller.

-pepper_dss\algorithms\RL_Algorithms\pepper_webots\controllers\supervisorController\supervisorControllerNAF.py is an environement used to interact with the NAF agent, to used this agent simply rename the file to supervisorController.py or modify the supervisor robots controller in Webots by pointing it to supervisorControllerNAF.py (The former is easier IMO). The NAF agent requires the supporting files; agent.py, networks.py, and replay_buffer.py.

-The PPO agent is also included, however this agent is inferior to the others and is not used. 


### Tutorials and Info

- Open AI Baselines:
  - https://github.com/openai/baselines

- Deepbots Webots Integration:
  - https://github.com/aidudezzz/deepbots

- Introduction to Webots:
  - https://cyberbotics.com/doc/guide/getting-started-with-webots


