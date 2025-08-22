from random import random, choice
from collections import defaultdict

from statistics import mean
import matplotlib.pyplot as plt

states = [0,1,2,3,4,5]
actions = [-1,1]

def model_distribution(from_state, action):

	state_probability = random()
	state_reached = -1

	if state_probability <= 0.8:
		state_reached =  from_state + action
	elif state_probability <= 0.95:
		state_reached = from_state
	else:
		state_reached = from_state - action

	return state_reached


def reward(current_robot_state,action_taken):

	if current_robot_state==0 and action_taken == -1:
		return 1

	elif current_robot_state==5 and action_taken == 1:
		return 5

	else:
		return 0

def policy(state, Q):
	actions = [-1,1]
	if Q[str(state)+','+str(1)] < Q[str(state)+','+str(-1)]:
		return -1
	elif Q[str(state)+','+str(1)] > Q[str(state)+','+str(-1)]:
		return 1
	else:
		return choice(actions)


def Value_estimate(state, Q):

	if Q[str(state)+','+str(1)] < Q[str(state)+','+str(-1)]:
		return Q[str(state)+','+str(-1)]
	elif Q[str(state)+','+str(1)] > Q[str(state)+','+str(-1)]:
		return Q[str(state)+','+str(1)]
	else:
		return (Q[str(state)+','+str(1)] + Q[str(state)+','+str(-1)])/2


V = [[] for i in range(0,6)]

Q = {}

Returns = {}

for s in states:
	for a in actions:
		#policy[str(a)+'|'+ str(s)] = 0.5
		Q[str(s)+','+str(a)] = 0
		Returns[str(s)+','+str(a)] = []


# Write a function to choose a state randomly and to choose and action at the state. Evolve from the state based on the chosen policy


#Iterating over episodes
episode_length = 6000
for i in range(0,episode_length):

	robot_state = choice(states[1:5])

	Rewards = []

	state_action_pairs = []

	time_steps = 0

	G = 0

	gamma = 0.95

	init = True

	if i%1000 ==0:
		print(i)

	while(True):

		if(robot_state == 0 or robot_state ==5):
			break

		
		if(init):
			action_taken = choice(actions)
			init = False
		else:
			action_taken = policy(robot_state,Q)


		#Based on model distribution enter a state
		state_reached = model_distribution(robot_state,action_taken)


		#### IMPORTANT
		#Reward is based on action AND reached state, that is reward for reaching for state 5 = 5 IFF action taken at state 4 is 1
		current_reward = reward(state_reached,action_taken)

		Rewards.append(current_reward)

		#Enter state, action pairs encountered during the episode in a list

		state_action_pairs.append(str(robot_state)+','+str(action_taken))

		robot_state = state_reached

		time_steps +=1

		# and repeat till either state ==0 or state ==5 hence resulting in end of episode

	if time_steps == 0:
		continue



	for j in reversed(range(time_steps)):

		G = gamma*G + Rewards[j] 

		#Condition to check if the state action pair at time t is not present in the list from 0 to t-1

		if  state_action_pairs[j] not in state_action_pairs[0:j] :

			# When the condition is satisfied update the value of G to Q(st,at)

			Returns[state_action_pairs[j]].append(G)
			Q[state_action_pairs[j]] = mean(Returns[state_action_pairs[j]])  

			# ^^^ Update policy as pi(a|s) = 1 is a = argmax_a Q(st,a) else 0 - This is the current policy

	for s in states:
		V[s].append(Value_estimate(s,Q))


print(Q)
for i in range(1,5):
	print("state", i)
	print("policy", policy(i,Q))



for s in states:
	if s!=0 and s!=5:
		plt.plot(V[s], label='state' + ' ' + str(s))

plt.xlabel("Episodes")
plt.ylabel("Value function")
plt.title("Value function at all the non-terminal states")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()

plt.show()