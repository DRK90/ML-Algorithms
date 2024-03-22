import pandas as pd
import numpy as num
import matplotlib.pyplot as plt

class RaceTrackModel:
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

class ValueLearner:
    def __init__(self, raceTrack, discountFactor, maxSpeed= 5):
        self.track = raceTrack
        self.maxSpeed = maxSpeed
        self.valueTable = num.zeros((raceTrack.track_height, raceTrack.track_width, 2*maxSpeed+1, 2*maxSpeed+1))
        self.discountFactor = discountFactor
        
    def reward(self, x, y):
        #0 for finish, -5 for crash, -1 for normal
        if self.track.track[y][x] == 'F':
            return 0
        elif self.track.track[y][x] == '#':
            return -5
        else:
            return -1
        
    def valueIteration(self, threshold=.001):
        #continue loop until convergence
        deltas = []
        counter = 0
        while True:
            counter += 1
            #track the change in value for each iteration
            delta = 0
            #iterate over every possible combo of track position and speed
            for y in range(self.track.track_height):
                for x in range(self.track.track_width):
                    for vx in range(-self.maxSpeed, self.maxSpeed+1):
                        for vy in range(-self.maxSpeed, self.maxSpeed+1):
                            ##get the current value from the table to compare
                            oldValue = self.valueTable[y, x, vx+self.maxSpeed, vy+self.maxSpeed]
                            #get the new value
                            newValue = self.calculateNewValue(x, y, vx, vy)
                            #update the value table with the new value calculated
                            self.valueTable[y, x, vx+self.maxSpeed, vy+self.maxSpeed] = newValue
                            #get if the new difference is greater than delta, then update it
                            delta = max(delta, abs(oldValue - newValue))
            deltas.append(delta)
            #compare delta to the threshold CONVERGENCE
            if delta < threshold:
                counters = range(1,counter+1)
                if 1==0:
                    plt.figure(figsize=(10,6))
                    plt.plot(counters, deltas, color='b', linestyle ='-')
                    plt.title("Delta vs. Iteration")
                    plt.xlabel("Iteration")
                    plt.ylabel("Delta")
                    plt.grid(True)
                    plt.savefig('/Users/dk/Documents/my_plots/VALUE_LAST_OTRACK.pdf')
                    #plt.show()
                break
    
    def calculateNewValue(self, x, y, vx, vy):
        #all possible action combinations
        possibleMoves = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1), (0, 0), (0, 1), 
                  (1, -1), (1, 0), (1, 1)]
        #init list for the value of each move
        values = []

        for ax, ay in possibleMoves:
            #stay within track limits, and get the new velocity
            newVx, newVy = min(max(vx + ax, -self.maxSpeed), self.maxSpeed), min(max(vy + ay, -self.maxSpeed), self.maxSpeed)
            #position + new velocity, get the new position
            newX, newY = min(max(x + newVx, 0), self.track.track_width-1), min(max(y + newVy, 0), self.track.track_height-1)
            #get the reward based on the new state
            reward = self.reward(newX, newY)
            #add the value (with discount factor applied) to the value list
            values.append(reward + self.discountFactor * self.valueTable[newY, newX, newVx+self.maxSpeed, newVy+self.maxSpeed])
        #return the maximum of the values found, if none work within bounds of the track, return 0
        return max(values) if values else 0
        
    def getPolicy(self):
        #get the best policy based on the learned value table
        #represent every possible position and velocity, to represent all states of the table
        policy = num.zeros((self.track.track_height, self.track.track_width, 2*self.maxSpeed+1, 2*self.maxSpeed+1, 2), dtype=int)
        #iterate over every possible state
        for y in range(self.track.track_height):
            for x in range(self.track.track_width):
                for vx in range(-self.maxSpeed, self.maxSpeed+1):
                    for vy in range(-self.maxSpeed, self.maxSpeed+1):
                        #get the best action based on the current value table, add the action to the correct state
                        policy[y, x, vx+self.maxSpeed, vy+self.maxSpeed] = self.getBestAction(x, y, vx, vy)
        return policy
    
    def getBestAction(self, x, y, vx, vy):
        #all possible action combinations
        possibleMoves = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1), (0, 0), (0, 1), 
                  (1, -1), (1, 0), (1, 1)]
        bestAction = None
        bestValue = float('-inf')
        #iterate over all the possible actions
        for ax, ay in possibleMoves:
            #STAY IN CONFINES OF TRACK - get the new velocity from the action iterated on
            newVx, newVy = min(max(vx + ax, -self.maxSpeed), self.maxSpeed), min(max(vy + ay, -self.maxSpeed), self.maxSpeed)
            #new position based on updated velocity
            newX, newY = min(max(x + newVx, 0), self.track.track_width-1), min(max(y + newVy, 0), self.track.track_height-1)
            #get the value from the value table
            value = self.valueTable[newY, newX, newVx+self.maxSpeed, newVy+self.maxSpeed]
            #update best value depending on the value calculated
            if value > bestValue:
                bestValue = value
                bestAction = (ax, ay)
        return bestAction



    
