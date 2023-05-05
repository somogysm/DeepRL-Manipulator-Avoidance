from deepbots.robots.controllers.robot_emitter_receiver_csv import RobotEmitterReceiverCSV
import numpy as np
import ast
from sklearn import preprocessing

class KinovaRobot(RobotEmitterReceiverCSV):
	"""Kinova robot Webots controller.

			Attributes:
				self.numMotors: Number of motors, Int.
				self.motors: List containing Webots motor objects.
				self.sensors: List containing Webots sensor objects.
				self.actionSpace: Number of actions, Int.
				self.robot: Webots robot object.
	"""
	def __init__(self):
		"""Inits the robot controller.

				Args:
					None

				Returns:
					None
		"""
		super().__init__()
		self.numMotors = 7
		self.motors = []
		self.sensors = []
		for i in range(self.numMotors):
			motor = self.robot.getDevice('joint ' + str(i+1) + ' motor')  # Get the wheel handle
			motor.setPosition(float('inf'))  # Set starting position
			motor.setVelocity(0.0)  # Zero out starting velocity
			self.motors.append(motor)

			positionSensor = self.robot.getDevice('joint ' + str(i+1) + ' sensor')
			positionSensor.enable(self.get_timestep())
			self.sensors.append(positionSensor)

	def create_message(self):
		"""Creates message to be transmitted to the supervisor.

				Args:
					None

				Returns:
					Desired message, containing list of robot observations.
		"""
		# Read the sensor value, convert to string and save it in a list
		# Read the sensor value, convert to string and save it in a list
		message1 = [self.sensors[i].getValue() for i in range(self.numMotors)]
		message3 = [np.sin(i) for i in message1]
		message4 = [np.cos(i) for i in message1]
		
		message2 = [self.motors[i].getVelocity() for i in range(self.numMotors)]
		return message2 + message3 + message4
	
	def use_message_data(self, message):
		"""Read incoming message recieved from the supervisor. (Needs to be modified to read/decipher incoming data depending on the agent.

				Args:
					message: Message recieved from the supervisor

				Returns:
					None
		"""
		#action = preprocessing.normalize(np.array(message).reshape(1, -1)).reshape(-1, 1) # Convert the string message into an action integer
		#action = message
		action = ast.literal_eval('[' + message[0].replace('             ', ',').replace('            ', ',').replace('           ', ',').replace('          ', ',').replace('         ', ',').replace('        ', ',').replace('       ', ',').replace('      ', ',').replace('     ', ',').replace('    ', ',').replace('   ', ',').replace('  ', ',').replace(' ',',').strip(']').strip('[').strip(',')+']')
		#print(action)
		for i in range(self.numMotors):
			self.motors[i].setVelocity(float(action[i])*1)
		
		# Set the motors' velocities based on the action received

if __name__ == "__main__":
	# Create the robot controller object and run it
	robot_controller = KinovaRobot()
	robot_controller.run()  # Run method is implemented by the framework, just need to call it
