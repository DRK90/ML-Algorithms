import pandas as pd
import numpy as num

class RaceTrackValueModel:
    def __init__(self, filePath):
        #load the track
        self.track, self.track_width, self.track_height = self.loadTrack(filePath)
        #set the start position
        self.carX, self.carY = self.findStartPosition()
        #initialize velocity to 0,0
        self.car_VX, self.car_VY = 0, 0
    
    def loadTrack(self, filePath):
        track = []
        with open(filePath, 'r') as file:
            # Read the first line to get dimensions
            dimensions = file.readline().strip().split(',')
            track_height, track_width = int(dimensions[0]), int(dimensions[1])

            # Load the rest of the track
            track = [list(line.strip()) for line in file]

        return track, track_width, track_height
    
    def findStartPosition(self):
        #start on an S
        for y, row in enumerate(self.track):
            for x, cell in enumerate(row):
                if cell == 'S':
                    return x, y
                
    def step(self, action):
            # get acceleration from the action
            ax, ay = action

            #maybe we dont accelerate
            if num.random.rand() > 0.8:  # 20% chance to maintain current velocity
                ax, ay = 0, 0  # Override the acceleration


            # update velocity based on accel {-5, 5}
            self.car_VX = min(max(self.car_VX + ax, -5), 5)
            self.car_VY = min(max(self.car_VY + ay, -5), 5)

            # Check each step from the current to the new position
            for i in range(1, max(abs(self.car_VX), abs(self.car_VY)) + 1):
                step_x = self.carX + int(i * self.car_VX / max(abs(self.car_VX), abs(self.car_VY)))
                step_y = self.carY + int(i * self.car_VY / max(abs(self.car_VX), abs(self.car_VY)))

                # Check if this step is within track boundaries
                if not (0 <= step_x < self.track_width and 0 <= step_y < self.track_height):
                    return self.handle_crash()

                # Check if this step is on the finish line
                if self.track[step_y][step_x] == 'F':
                    self.carX, self.carY = step_x, step_y
                    return (self.carX, self.carY, self.car_VX, self.car_VY), 0, True  # Finished

                # Check if this step is a crash
                if self.track[step_y][step_x] == '#':
                    return self.handle_crash()

            # Update to final position
            self.carX += self.car_VX
            self.carY += self.car_VY

            # Normal step cost
            return (self.carX, self.carY, self.car_VX, self.car_VY), -1, False

    def handle_crash(self):
        # Reset to start position
        self.carX, self.carY = self.findStartPosition()
        self.car_VX = 0
        self.car_VY = 0
        return (self.carX, self.carY, self.car_VX, self.car_VY), -5, False
    
    def handle_crash_last_valid(self):
        #return to the last position, without moving the car.  reset velocity to 0.
        #print("***CRASH***")
        self.car_VX = 0
        self.car_VY = 0
        return (self.carX, self.carY, 0, 0), -5, False
    
class Learner:
    def __init__(self, numStates, numActions, learningRate, discountFactor, epsilon):
        #initialize q table with zeros
        self.qTable = num.zeros((numStates, numActions))
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.numStates = numStates
        self.numActions = numActions

    def chooseAction(self, state):
        #exploration at first, then exploitation.
        if num.random.rand() < self.epsilon:
            #return random action
            return num.random.choice(self.numActions) 
        #or the best known action
        return num.argmax(self.qTable[state])
    
    #learn for QLEARN
    def learn(self, state, action, reward, nextState):
        #update rule for q-learning, the next state at the crash sequence is always the same, so that impacts the tdTarget
        nextBestAction = num.argmax(self.qTable[nextState])
        tdTarget = reward + self.discountFactor * self.qTable[nextState][nextBestAction]
        tdError = tdTarget - self.qTable[state][action]
        #here, if the action did nothing, we shouldnt update anything TO DO OR NOT actually
        self.qTable[state][action] += self.learningRate * tdError

    #learn for SARSA
    def Sarsalearn(self, state, action, reward, nextState, nextAction):
        #SARSA, include the next action in updating
        tdTarget = reward + self.discountFactor * self.qTable[nextState][nextAction]
        tdError = tdTarget - self.qTable[state][action]
        self.qTable[state][action] += self.learningRate * tdError
    
