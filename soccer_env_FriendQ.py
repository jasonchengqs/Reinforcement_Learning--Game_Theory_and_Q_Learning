from copy import copy
import numpy as np
import random

alpha_max = 1;
alpha_min = 0.001;
decay = 0.999995;
gemma = 0.9;
epsilon_max = 0.01;
# epsilon_decay = 0.993;

class Agent:

	def __init__(self, x, y, ball=None, S=None, actions=None, agent_id=None):
		self.x = x;
		self.y = y;
		self.ball = ball;
		self.agent_id = agent_id;
		self.actions = actions;
		self.alpha = alpha_max;
		self.epsilon = epsilon_max;
		self.Q = {};
		if S != None:
			for s in S:
				for a in actions:
					for o in actions:
						self.Q[(s,a,o)] = 1;
		self.step = 1;
		#print("Q---",self.Q)
		#print ("keys----",self.Q.keys())

	def update(self, field, arg1, arg2=None, arg3=None):
		if field == 'x':
			self.x = arg1;
		if field == 'y':
			self.y = arg1;
		if field == 'ball':
			self.ball = arg1;

		if field == 'all':
			self.x = arg1;
			self.y = arg2;
			self.ball = arg3;

	def Q_learn(self, reward, action, is_score, s, s_next):
		a = action[self.agent_id];
		o_id = set(action.keys())-set(self.agent_id);
		o = action[o_id.pop()];
		r = reward[self.agent_id];
		if not is_score:
			nextlist = []
			for k in list(self.Q.keys()):
				if s_next in k:
					nextlist.append(self.Q[k])
			# print ("nextlist:", nextlist, len(nextlist))
			self.Q[(s,a,o)] += self.alpha*(r + gemma*np.amax(nextlist) - self.Q[(s,a,o)])
		else:
			self.Q[(s,a,o)] += self.alpha*(r - self.Q[(s,a,o)])
		
		if self.alpha > alpha_min:
			# self.alpha = 1/self.step;
			self.alpha = self.alpha * decay;

	def actionSelect(self, s):
		# print("s:", s)
		if random.random() < self.epsilon:
			action = random.choice(self.actions);
		else:
			slist = []
			alist = []
			for k in list(self.Q.keys()):
				# print ('k---', k)
				if s in k:
					slist.append(self.Q[k]);
					alist.append(k[1])
			action_id = np.argmax(slist);
			action = alist[action_id];

		# self.epsilon = self.epsilon*epsilon_decay;
		# print (slist)
		# print (alist)
		print (self.agent_id, 'actionSelect:', action)
		return action

class Game:

	def __init__(self, n_row, n_col, agents, goal, goal_col):
		self.n_col = n_col;
		self.n_row = n_row;
		self.goal = goal;
		self.goal_col = goal_col;
		
		self.agents = {};
		for agent in agents:
			self.agents[agent.agent_id] = agent;
		print(agents)
		print (self.agents)

	def move(self, action):

		agent_order = np.random.permutation(list(action.keys()));
		# print (agent_order)
		cur_agent = Agent(x=0, y=0, ball=False);
		for agent_id in agent_order:
			mover = self.agents[agent_id];
			others_id = set(agent_order)-set(agent_id);
			others = self.agents[others_id.pop()];

			cur_agent.update('all', mover.x, mover.y, mover.ball);
			a = action[agent_id];
			if a == 'N' and cur_agent.y != 0:
				cur_agent.update('y', cur_agent.y - 1);
			elif a == 'E' and cur_agent.x < self.n_col - 1:
				cur_agent.update('x', cur_agent.x + 1);
			elif a == "W" and cur_agent.x > 0:
				cur_agent.update('x', cur_agent.x - 1);
			elif a == 'S' and cur_agent.y != self.n_row - 1:
				cur_agent.update('y', cur_agent.y + 1);

			is_collision = self.collide(cur_agent, mover, others);

			if is_collision == False:
				mover.update('all', cur_agent.x, cur_agent.y, cur_agent.ball);

			r, is_score = self.score(mover, others);

			if is_score:
				break;

		for a_id in self.agents.keys():
			self.agents[a_id].step += 1;

		return r, is_score, 


	def collide(self, cur_agent, mover, others):

		is_collision = False;

		if cur_agent.x == others.x and cur_agent.y == others.y:

			is_collision = True;
			print('collide--')
			if cur_agent.ball:
				mover.update('ball', False);
				others.update('ball', True);
				print('ball possestion switched')
		
		return is_collision
				

	def score(self, mover, others):
		score = False;
		reward = {x: 0 for x in self.agents.keys()}
		if mover.x == self.goal_col[mover.agent_id] and mover.ball == True:
			print (mover.agent_id, "score!")
			score = True;
			for r_id in reward.keys():
				reward[r_id] = -self.goal;
			reward[mover.agent_id] = self.goal;
		else:
			own = {i: mover.x == self.goal_col[i] and mover.ball \
			for i in self.agents.keys()};
			if sum(own.values()) > 0:
				print(mover.agent_id, "score own goal!")
				score = True;

				for r_id in reward.keys():
					reward[r_id] = self.goal;
				reward[mover.agent_id] = -self.goal;

		if others.x == self.goal_col[others.agent_id] and others.ball == True:
			print (others.agent_id, "score!")
			score = True;
			for r_id in reward.keys():
				reward[r_id] = -self.goal;
			reward[others.agent_id] = self.goal;
		else:
			own = {i: others.x == self.goal_col[i] and others.ball \
			for i in self.agents.keys()};
			if sum(own.values()) > 0:
				print(others.agent_id, "score own goal!")
				score = True;

				for r_id in reward.keys():
					reward[r_id] = self.goal;
				reward[others.agent_id] = -self.goal;
		return reward, score
	
	def reset(self):
		for a_id in self.agents.keys():
			if self.agents[a_id].agent_id == 'A':
				self.agents[a_id].update('all', 2, 0, False);
			if self.agents[a_id].agent_id == 'B':
				self.agents[a_id].update('all', 1, 0, True);


