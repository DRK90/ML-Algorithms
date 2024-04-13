import RaceTrack as rt
import time
import matplotlib.pyplot as plt

trackFilePath = "./race/L-track-1.txt"
raceTrack = rt.RaceTrackModel(trackFilePath)
maxVelocityRange = 11 #-5 through 5, 11 options
#numStates = vx * vx * height * width
numStates = maxVelocityRange * maxVelocityRange * raceTrack.track_height * raceTrack.track_width
#numActions = all combos of acceleration 9
numActions = 9
#learn class for q and sarsa
agent = rt.Learner(numStates, numActions, 0.1, .8, 0.99) #..learning Rate, Discount Factor, Epsilon
#value learning for value iteration
#valueLearner = rt.ValueLearner(raceTrack, discountFactor=.8)
#value iteration first
#valueLearner.valueIteration(threshold=.001)
#get the optimal policy
#optimalPolicy = valueLearner.getPolicy()


def encodeState(state):
    x, y, vx, vy = state
    maxVelocity = 5
    #encode velocity by shifting the negative values
    encodedVx = vx + maxVelocity
    encodedVy = vy + maxVelocity
    #get track width and height
    trackWidth = raceTrack.track_width
    #get the state index 
    return (y * trackWidth + x) * (2 * maxVelocity + 1) ** 2 + encodedVy * (2 * maxVelocity + 1) + encodedVx

def decodeAction(actionIndex):
    # Map the action index to a combination of ax and ay
    action_map = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1), (0, 0), (0, 1), 
                  (1, -1), (1, 0), (1, 1)]
    return action_map[actionIndex]

def printTrack(raceTrack):
    for y, row in enumerate(raceTrack.track):
        for x, cell in enumerate(row):
            if x == raceTrack.carX and y == raceTrack.carY:
                print('C', end='')  # Representing the car
            else:
                print(cell, end='')
        print()

#iterate through
i = 1000
counters = []
#for Q LEARNING
if 1==0:
    for loop in range(i):
        raceTrack.carX, raceTrack.carY = raceTrack.findStartPosition()
        raceTrack.car_VX, raceTrack.car_VY = (0,0)
        state = (raceTrack.carX, raceTrack.carY, raceTrack.car_VX, raceTrack.car_VY)
        stateIndex = encodeState(state)

        done = False
        counter = 0
        #after 10,000 iterations, epsilon will be 0, so we stop exploring new routes, and only take the best route.
        agent.epsilon -= .0001
        while not done:        
            counter += 1
            if counter == 700:
                print("stop")
            
            #choose action
            actionIndex = agent.chooseAction(stateIndex)
            action = decodeAction(actionIndex)

            #apply the action
            newState, reward, done = raceTrack.step(action)
            #if done == True and counter < 40:
                #print("No way")
            newStateIndex = encodeState(newState)

            # apply the state to learn
            agent.learn(stateIndex, actionIndex, reward, newStateIndex)

            # Move to the new state
            stateIndex = newStateIndex

            #if loop >= 0 and 1==1:
            #   printTrack(raceTrack)
            #  print(f"XSpeed: {raceTrack.car_VX}, YSpeed: {raceTrack.car_VY}")
                #time.sleep(.01)
        print(counter)
        #print(raceTrack.car_VX)
        #print(raceTrack.car_VY)
        counters.append(counter)

#for SARSA
if 1==1:
    for loop in range(i):
        raceTrack.carX, raceTrack.carY = raceTrack.findStartPosition()
        raceTrack.car_VX, raceTrack.car_VY = (0,0)
        state = (raceTrack.carX, raceTrack.carY, raceTrack.car_VX, raceTrack.car_VY)
        stateIndex = encodeState(state)
        actionIndex = agent.chooseAction(stateIndex)


        done = False
        counter = 0
        #after 10,000 iterations, epsilon will be 0, so we stop exploring new routes, and only take the best route.
        agent.epsilon -= .001
        while not done:        
            counter += 1

            #decrease epsilon for R TRACK
            #print(counter)
            
            #choose action
            action = decodeAction(actionIndex)

            #apply the action
            newState, reward, done = raceTrack.step(action)
            #if done == True and counter < 40:
                #print("No way")
            newStateIndex = encodeState(newState)

            nextActionIndex = agent.chooseAction(newStateIndex)

            # apply the state to learn
            agent.Sarsalearn(stateIndex, actionIndex, reward, newStateIndex, nextActionIndex)

            # Move to the new state
            stateIndex = newStateIndex
            actionIndex = nextActionIndex

            #if loop >= 0 and 1==1:
            #   printTrack(raceTrack)
            #  print(f"XSpeed: {raceTrack.car_VX}, YSpeed: {raceTrack.car_VY}")
                #time.sleep(.01)
        print(counter)
        #print(raceTrack.car_VX)
        #print(raceTrack.car_VY)
        counters.append(counter)

#for Value Iteration
if 1==1:
    for loop in range(10):
        raceTrack.carX, raceTrack.carY = raceTrack.findStartPosition()
        raceTrack.car_VX, raceTrack.car_VY = (0,0)
        state = (raceTrack.carX, raceTrack.carY, raceTrack.car_VX, raceTrack.car_VY)

        done = False
        counter = 0
        #after 10,000 iterations, epsilon will be 0, so we stop exploring new routes, and only take the best route.
        agent.epsilon -= .001
        while not done:        
            counter += 1

            #decrease epsilon for R TRACK
            #print(counter)
            
            #choose action
            bestAction = optimalPolicy[raceTrack.carY, raceTrack.carX, raceTrack.car_VX + valueLearner.maxSpeed, raceTrack.car_VY + valueLearner.maxSpeed]

            #apply the action
            newState, reward, done = raceTrack.step(bestAction)

            raceTrack.carX, raceTrack.carY, raceTrack.car_VX, raceTrack.car_VY = newState



            #if loop >= 0 and 1==1:
             #  printTrack(raceTrack)
              # print(f"XSpeed: {raceTrack.car_VX}, YSpeed: {raceTrack.car_VY}")
               #time.sleep(.01)
        print(counter)
        #print(raceTrack.car_VX)
        #print(raceTrack.car_VY)
        counters.append(counter)
        print(sum(counters)/len(counters))

if 1==0:
    iterations = range(1,i+1)
    plt.figure(figsize=(10,6))
    plt.plot(iterations, counters, color='b', linestyle ='-')
    plt.title("Steps vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.savefig('/Users/dk/Documents/my_plots/VALUE_RESTART_RTRACK.pdf')
    plt.show()




