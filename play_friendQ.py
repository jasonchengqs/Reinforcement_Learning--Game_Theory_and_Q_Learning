from soccer_env_FriendQ import Agent, Game
from copy import deepcopy
import math
import matplotlib.pyplot as plt

def main():
	n_row = 2; n_col = 4;
	n_states = n_row * n_col;
	goal = 100;
	goal_col = {'A':0, 'B':3};
	actions = ['N', 'S', 'E', 'W', 'K'];
	ball = ['A', 'B'];
	S = [];
	for b in ball:
		for r_B in range(n_row):
			for c_B in range(n_col):
				for r_A in range(n_row):
					for c_A in range(n_col):
						if (r_A != r_B) or (c_A != c_B):
							S.append((c_A, r_A, c_B, r_B, b));

	agent_A = Agent(2, 0, False, S, actions, agent_id='A');
	agent_B = Agent(1, 0, True, S, actions, agent_id='B');
	agents = [agent_A, agent_B];

	game=Game(n_row, n_col, agents, goal, goal_col);

	T = 10000;

	error_list = [];
	for t in range(T):
		old_Q = deepcopy(agent_A.Q);
		# print ("1----oldQ", sum(old_Q.values()));
		if agent_A.ball:
			b = 'A';
		else:
			b = 'B';
		s = (deepcopy(agent_A.x), deepcopy(agent_A.y), \
			deepcopy(agent_B.x), deepcopy(agent_B.y), deepcopy(b));
		
		a_A = agent_A.actionSelect(s);
		a_B = agent_B.actionSelect(s);
		action = {'A': a_A, 'B': a_B};
		
		
		reward, is_score = game.move(action);
		if agent_A.ball:
			b = 'A';
		else:
			b = 'B';
		s_next = (agent_A.x, agent_A.y, agent_B.x, agent_B.y, b)

		agent_A.Q_learn(reward, action, is_score, s, s_next);
		agent_B.Q_learn(reward, action, is_score, s, s_next);
		print('reward:', reward);
		print (sum(agent_A.Q.values()), sum(agent_B.Q.values()))
		# print ("2----oldQ", sum(old_Q.values()));
		# print ("2----AQ", sum(agent_A.Q.values()));
		err = 0;
		for k in list(old_Q.keys()):
			if k[1] == 'S' and k[2] == 'K':
				err += (agent_A.Q[k] - old_Q[k])**2;
		err = math.sqrt(err);
		print('err:', err);
		error_list.append(err);

		if is_score:
			game.reset();

	#action = {'A':'K', 'B':'W'};
	#reward, score = game.move(action);

	# print ('A-x:',agent_A.x, 'y:', agent_A.y, 'ball:', agent_A.ball)
	# print ('B-x:',agent_B.x, 'y:', agent_B.y, 'ball:', agent_B.ball)
	# print (reward)
	# print ("error:", error_list)

	plt.plot([t for t in range(T)], error_list);
	plt.xlabel('steps')
	plt.ylabel('Difference of Q')
	plt.show()

if __name__ == '__main__':
	main()