#!/usr/bin/env python3
from concurrent.futures import process
import vizdoom as vzd
import os
import random
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import skimage.transform
from time import time, sleep
from tqdm import trange
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

if torch.cuda.is_available:
  DEVICE = torch.device('cuda')
  print("[+] Using CUDA")
else:
  DEVICE = torch.device('cpu')
  print("[+] Using CPU")

config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")
model_path = "models/doomguy.pth"
save_model = True

replay_memory_size = 10000
discount_factor = 0.99
load_model = False

learning_rate = 1e-4
batch_size = 64
#resolution = (128, 128)
resolution = (30, 45)
n_epochs = 100

frame_repeat = 12
learning_steps_per_epoch = 2000

skip_learning = False
test_episodes_per_epoch = 100
episodes_to_watch = 10

class DQN(nn.Module):
  def __init__(self, n_avail_actions):
    super(DQN, self).__init__()

    self.conv1 = nn.Sequential(
      nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    )

    self.conv2 = nn.Sequential(
      nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    )
    
    self.conv3 = nn.Sequential(
      nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(8),
      nn.ReLU()
    )

    self.conv4 = nn.Sequential(
      nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
      nn.BatchNorm2d(16),
      nn.ReLU()
    )

    self.state_fc = nn.Sequential(
      nn.Linear(96, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

    self.advantage_fc = nn.Sequential(
      nn.Linear(96, 64),
      nn.ReLU(),
      nn.Linear(64, n_avail_actions)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = x.view(-1, 192) # TODO: num_flat_feats()
    x1 = x[:, :96]  # input for the net to calculate the state value
    x2 = x[:, 96:]  # relative advantage of actions in the state
    state_value = self.state_fc(x1).reshape(-1, 1)
    advantage_values = self.advantage_fc(x2)
    x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

    return x


class DoomGuyAgent:
  def __init__(self, action_size, memory_size, batch_size, discount_factor,
                lr, load_model, epsilon=1, epsilon_decay=0.9996, epsilon_min=0.1):
    self.action_size = action_size
    self.epsilon = epsilon
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min
    self.batch_size = batch_size
    self.discount = discount_factor
    self.lr = lr
    self.memory = deque(maxlen=memory_size)
    self.criterion = nn.MSELoss()

    if load_model:
      self.q_net = torch.load(model_path)
      self.target_net = torch.load(model_path)
      self.epsilon = epsilon_min
    else:
      print("[+] Creating New Deep Learning Model ...")
      self.q_net = DQN(action_size).to(DEVICE)
      self.target_net = DQN(action_size).to(DEVICE)

    self.opt = optim.SGD(self.q_net.parameters(), lr=self.lr)

  def get_action(self, state):
    if np.random.uniform() < self.epsilon:
      # exploration
      return random.choice(range(self.action_size))
    else:
      # exploitation
      state = np.expand_dims(state, axis=0)
      state = torch.from_numpy(state).float().to(DEVICE)
      action = torch.argmax(self.q_net(state)).item()
      return action

  def update_target_net(self):
    self.target_net.load_state_dict(self.q_net.state_dict())

  def append_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def train(self):
    batch = random.sample(self.memory, self.batch_size)
    batch = np.array(batch, dtype=object)

    states = np.stack(batch[:, 0]).astype(float)
    actions = np.stack(batch[:, 1]).astype(float)
    rewards = np.stack(batch[:, 2]).astype(float)
    next_states = np.stack(batch[:, 3]).astype(float)
    dones = batch[:, 4].astype(bool)
    not_dones = ~dones

    row_idx = np.arange(self.batch_size)  # indexes the batch

    with torch.no_grad():
      next_states = torch.from_numpy(next_states).float().to(DEVICE)
      idx = row_idx, np.argmax(self.q_net(next_states).cpu().data.numpy(), 1)
      next_state_values = self.target_net(next_states).cpu().data.numpy()[idx]
      next_state_values = next_state_values[not_dones]

    # y = r + discount * max_a q(s', a)
    q_targets = rewards.copy()
    q_targets[not_dones] += self.discount * next_state_values
    q_targets = torch.from_numpy(q_targets).float().to(DEVICE)

    # select only q values of the actions taken
    idx = row_idx, actions
    states = torch.from_numpy(states).float().to(DEVICE)
    action_values = self.q_net(states)[idx].float().to(DEVICE)

    self.opt.zero_grad()
    td_error = self.criterion(q_targets, action_values)
    td_error.backward()
    self.opt.step()

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
    else:
      self.epsilon - self.epsilon_min


def game_init():
  print("[+] Initializing Doom ...")
  game = vzd.DoomGame()
  game.load_config(config_file_path)
  game.set_window_visible(False)
  game.set_mode(vzd.Mode.PLAYER)
  game.set_screen_format(vzd.ScreenFormat.GRAY8)
  game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
  game.init()
  print("[+] Game Initialized")
  return game

def process_frame(img):
  #img = cv2.resize(img, resolution)
  img = skimage.transform.resize(img, resolution)
  img = img.astype(np.float32)
  img = np.expand_dims(img, axis=0)
  return img

def test(game, agent):
  print("Testing Agent ...")
  test_scores = []
  for test_episode in (t := trange(test_episodes_per_epoch)):
    game.new_episode()
    while not game.is_episode_finished():
      state = process_frame(game.get_state().screen_buffer)
      best_action_idx = agent.get_action(state)
      game.make_action(actions[best_action_idx], frame_repeat)

    r = game.get_total_reward()
    test_scores.append(r)

  test_scores = np.array(test_scores)
  print("Test Results: mean %.1f +/- %.1f, min: %.1f, max: %.1f"%(test_scores.mean(), test_scores.std(), test_scores.min(), test_scores.max()))

def run(game, agent, actions, n_epochs, frame_repeat, steps_per_epoch=2000):
  # for each epoch, for each episode, skip frame_repeat number of frames after each action
  start_time = time()

  try:
    for epoch in range(n_epochs):
      game.new_episode()
      train_scores = []
      global_step = 0
      print("[+] Epoch %d"%(epoch+1))

      for i in (t := trange(steps_per_epoch)):
        state = process_frame(game.get_state().screen_buffer)
        action = agent.get_action(state)
        reward = game.make_action(actions[action], frame_repeat)
        done = game.is_episode_finished()

        if not done:
          next_state = process_frame(game.get_state().screen_buffer)
        else:
          next_state = np.zeros((1, resolution[0], resolution[1]))
        
        agent.append_memory(state, action, reward, next_state, done)

        if global_step > agent.batch_size:
          agent.train()
        
        if done:
          train_scores.append(game.get_total_reward())
          game.new_episode()

        global_step += 1
      agent.update_target_net()
      train_scores = np.array(train_scores)
      print("Train Results: mean: %.1f +/- %.1f,"%(train_scores.mean(), train_scores.std()),
            "min %.1f,"%train_scores.min(), "max: %.1f"%train_scores.max())

      test(game, agent)
      if save_model:
        print("Saving model to:", model_path)
        torch.save(agent.q_net, model_path)
      print("Total time elapsed: %.2f minutes"%((time() - start_time) / 60.0))
  except KeyboardInterrupt:
    print("[-] Training was interrupted by user")
    print("[+] Saving model to:", model_path)
    torch.save(agent.q_net, model_path)
  
  game.close()
  return agent, game


if __name__ == '__main__':
  game = game_init()
  n = game.get_available_buttons_size()
  actions = [list(a) for a in it.product([0, 1], repeat=n)]

  agent = DoomGuyAgent(len(actions), lr=learning_rate, batch_size=batch_size,
                      memory_size=replay_memory_size, discount_factor=discount_factor,
                      load_model=load_model)
  if not skip_learning:
    print("[+] Training Agent ...")
    agent, game = run(game, agent, actions, n_epochs=n_epochs, frame_repeat=frame_repeat,
                      steps_per_epoch=learning_steps_per_epoch)
    print("[+] Training Done!")
  
  print("[+] Agent's live gameplay ...")
  game.close()
  game.set_window_visible(True)
  game.set_mode(vzd.Mode.ASYNC_PLAYER)
  game.init()

  for e in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
      state = process_frame(game.get_state().screen_buffer)
      best_action_idx = agent.get_action(state)

      # This makes the animation smoother when compared to make_action()
      game.set_action(actions[best_action_idx])
      for i in range(frame_repeat):
        game.advance_action()

    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)
