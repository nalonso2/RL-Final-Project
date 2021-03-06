""" Define controller """
import torch
import torch.nn as nn
import numpy as np
import random

from collections import namedtuple
from replay import ExperienceReplay
# from controller import Actor, Critic, DiscreteActor

from torch.distributions import Categorical



Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class RandomAgent(nn.Module):
    def __init__(self, possible_actions=None, device=None):
        super().__init__()


class RLAgent(nn.Module):
    def __init__(self, possible_actions=None, device=None):
        super().__init__()
        self.replay = ExperienceReplay()
        self.discounted_factor = 0.98
        self.device = device
        self.possible_actions = np.array([
            [-1.0, 0., 0.],
            [-.75, 0., 0.],
            [-.5, 0., 0.],
            [-.25, 0., 0.],
            [.25, 0., 0.],
            [.5, 0., 0.],
            [.75, 0., 0.],
            [1.0, 0., 0.],
            [0.5, 0., 0.],
            [0., .33, 0.],
            [0., 0., .5],
        ]) if possible_actions is None else possible_actions
        self.n_actions = len(self.possible_actions)
        self.gs = 0
    
    def update(self):
        pass
    
    def get_action(self):
        pass
    
    def replay2batch(self, transitions):
        batch = Transition(*zip(*transitions))
        states = torch.cat(batch.state, dim=0).to(self.device)
        next_states = torch.cat([s for s in batch.next_state if s is not None], dim=0).to(self.device)
        actions = torch.cat(batch.action, dim=0).to(self.device)
        rewards = torch.stack(batch.reward, dim=0).to(self.device)
        return states, actions, rewards, next_states

class A3C(RLAgent):
    def __init__(self, input_dims, hidden_dims, memory_model, lr=1e-4, device=None):
        super(A3C, self).__init__(device=device)
        # self.n_actions = len(self.possible_actions) # Discrete version

        '''
        #########################################################
        INITIALIZE MEMORY UNIT HERE.
        
        Two options for implementation:
        1. Integrate unit directly into feature_extractor, 
        then don't have to update predict_v and predict_p.
        2. Initialize seperately from feature_extractor 
        then add to predict_v and predict_p.

        Will need to extract the lstm hidden state from
        the vector fed to feature_extractor since this vector will
        include both hidden state from LSTM and z from VAE. Looks
        like hidden state from LSTM is second half of "state" vector fed
        into predict_v and predict_p. See train-a3c.py around line 110.
        #########################################################
        '''
        #Initializes Memory module
        self.memory = memory_model

        #First layer of actor and critic NNs. Feeds into both.
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
        )


        #Input taken from feature extracter. Output are predicted q-values for actions
        self.actor = nn.Linear(hidden_dims, self.n_actions)

        #Input taken from feature extracter. Output is a scalar representing predicted value
        self.critic = nn.Linear(hidden_dims, 1)

        #Loss similar to MSE 
        self.vloss = nn.SmoothL1Loss()


    #These are outputs of critic which attempts to compute value of
    #next state
    def predict_v(self, state):
        '''
        Args
            state: (B, d)
        Return:
            v: (B, 1) tensor
        '''
        s = state.cpu()
        s_posterior = self.memory(s, mode=2)
        feature = self.feature_extractor(s_posterior)
        v = self.critic(feature)
        return v
    
    #Outputs of actor, which compute probabilities of taking certain actions
    #given current state.
    def predict_p(self, state):
        '''
        Args
            state: (B, d)
        Return:
            p: (B, n_actions) tensor
        '''
        s = state.cpu()
        s_posterior = self.memory(s, 1)
        feature = self.feature_extractor(s_posterior)
        p = torch.softmax(self.actor(feature), dim=-1)
        return p


    #Select action. Draw from probability distribution outputted by actor.
    def select_action(self, state):
        state.cpu()
        policy = self.predict_p(state) # (1, n_actions)
        m = Categorical(policy)
        a = m.sample() # (1,)
        action = self.possible_actions[a.item()] # (3, ) or (n_actions)
        p = policy.gather(1, a.unsqueeze(-1))
        return action, p # (3,) np array, (n_actions,) tensor



    '''
    Update function uses a recording of past (state, policy, rewards, prednextstate)
    to compute loss over actor outputs and critic outputs.
    Recording is similar to a minibatch which is averaged over. 
    '''
    def update(self, done):
        mem = self.replay.get_memory()
        states, policies, rewards, next_states = self.replay2batch(mem)

        #Target value for critic is cumulative discounted future reward
        #averaged over all possible actions.
        #This is estimated by looping backward through recording of rewards. 
        target = 0. # if done else self.predict_v(next_states[-1:]).item()
        targets = torch.zeros_like(rewards) # (B,)
        for idx in range(len(rewards)):
            target = rewards[-idx-1] + self.discounted_factor*target
            targets[-idx-1] = target


        #Outputs of critic for each state in batch
        V = self.predict_v(states)


        #A measures how much better actual reward was over predicted avg cum. future reward.
        #Actor attempts to maximize A.
        A = targets - V

        #Loss measures how well critic predicts cumulative future reward
        #and how rewarding actor's actions (predictions) tend to be.
        loss = self.pgloss(policies, A.detach()) + self.vloss(V, targets)
        
        loss.backward()
        self.gs += 1
        self.replay.reset()
    
    def pgloss(self, action_prob, reward):
        loss = -torch.mean(torch.log(action_prob+1e-6)*reward)
        return loss



def random_policy_noise(device):
    l, r = np.random.normal(0.0, 0.2, size=[1]), np.random.normal(-0.0, 0.2, size=[1])
    f = np.random.normal(0.6, 0.2, size=[1])
    return torch.tensor(np.concatenate([l, f, r]), dtype=torch.float).to(device)





"""
class REINFORCE(RLAgent):
    def __init__(self, input_dims, hidden_dims, lr=1e-4, possible_actions=None, device=None):
        super(REINFORCE, self).__init__(possible_actions=possible_actions, device=device)
        # self.n_actions = 2
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dims, self.n_actions)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)

    def select_action(self, state):
        feature = self.feature_extractor(state)
        policy = torch.softmax(self.actor(feature), dim=-1)
        # p = policy.detach().cpu().numpy().squeeze()
        # action = np.random.choice(self.n_actions, 1, p=p)
        # action = self.possible_actions[action]

        # policy = torch.sigmoid(self.actor(feature))
        # p = policy.detach().cpu().numpy().squeeze()
        m = Categorical(policy)
        a = m.sample()
        action = self.possible_actions[a.item()]
        # print(policy[0][a.item()])
        
        # action = np.asarray(p > 0.5, np.int)
        return action, policy[0][a.item()]

    def update(self, score=None):
        mem = self.replay.get_memory()
        R = 0.
        for s, a, r, ns in mem[::-1]:
            R = r + self.discounted_factor*R
            loss = self.pgloss(a, R)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        self.gs += 1
        self.replay.reset()

    def pgloss(self, action_prob, reward):
        loss = -torch.mean(torch.log(action_prob+1e-6)*reward)
        return loss


class AAC(RLAgent):
    def __init__(self, input_dims, hidden_dims, lr=1e-4, device=None):
        super(AAC, self).__init__(device=device)
        # self.n_actions = len(self.possible_actions) # Discrete version
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dims, self.n_actions)
        self.critic = nn.Linear(hidden_dims, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.vloss = nn.SmoothL1Loss()

    def predict_v(self, state):
        '''
        #Args
         #   state: (B, d)
        #Return:
         #   v: (B, 1) tensor
        '''
        feature = self.feature_extractor(state)
        v = self.critic(feature)
        return v
    
    def predict_p(self, state):
        '''
        #Args
        #   state: (B, d)
        #Return:
        #   p: (B, n_actions) tensor
        '''
        feature = self.feature_extractor(state)
        p = torch.softmax(self.actor(feature), dim=-1)
        return p

    def select_action(self, state):
        policy = self.predict_p(state) # (1, n_actions)
        m = Categorical(policy)
        a = m.sample() # (1,)
        action = self.possible_actions[a.item()] # (3, ) or (n_actions)
        p = policy.gather(1, a.unsqueeze(-1))
        return action, p # (3,) np array, (n_actions,) tensor

    def update(self):
        mem = self.replay.get_memory()
        states, policies, rewards, next_states = self.replay2batch(mem)
        target = 0.
        # targets = []
        targets = torch.zeros_like(rewards) # (B,)
        # print(rewards)
        for idx in range(len(rewards)):
            target = rewards[-idx-1] + self.discounted_factor*target
            targets[-idx-1] = target
            # targets.append([target])
        # targets.reverse()
        # targets = torch.tensor(targets, dtype=torch.float)

        V = self.predict_v(states)
        A = targets - V
        # policy = self.predict_p(states)
        # P = policy.gather(1, actions)
        loss = self.pgloss(policies, A.detach()) + self.vloss(V, targets)
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.gs += 1
        self.replay.reset()
    
    def pgloss(self, action_prob, reward):
        loss = -torch.mean(torch.log(action_prob+1e-6)*reward)
        return loss



class DDPG(RLAgent):
    def __init__(self, input_dims, hidden_dims, n_replays=10000, batch_size=32, tau=0.01, actor_lr=1e-4, critic_lr=5e-4, device=None):
        super(DDPG, self).__init__(n_replays=n_replays, batch_size=batch_size, device=device)
        self.actor = Actor(input_dims, hidden_dims, self.n_actions).to(device)
        self.critic = Critic(input_dims+self.n_actions, hidden_dims, self.n_actions).to(device)
        self.target_actor = Actor(input_dims, hidden_dims, self.n_actions).to(device)
        self.target_critic = Critic(input_dims+self.n_actions, hidden_dims, self.n_actions).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.a_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mse_loss = nn.MSELoss()
        self.tau = tau
    
    def select_action(self, state):
        # state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()

    def update(self):
        transitions = self.replay.sample(self.batch_size)
        states, actions, rewards, next_states = self.replay2batch(transitions)

        # Training Critic
        Q = self.critic(torch.cat([states, actions], dim=1))
        next_actions = self.target_actor(next_states).detach()
        next_Q = self.target_critic(torch.cat([next_states, next_actions], dim=1))
        y = rewards.unsqueeze(1) + self.discounted_factor*next_Q
        critic_loss = self.mse_loss(Q, y)

        self.c_optim.zero_grad()
        critic_loss.backward()
        self.c_optim.step()

        # Training Actor
        actions = self.actor(states)
        Q_pi = self.critic(torch.cat([states, actions], dim=1))
        actor_loss = -torch.mean(Q_pi)

        self.a_optim.zero_grad()
        actor_loss.backward()
        self.a_optim.step()

        self.param2target()
    
    def param2target(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

"""
