#!/usr/bin/env python3
import vizdoom as vzd
import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
from time import time, sleep
from tqdm import trange
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

config_file_path = os.path.join(vzd.scenarios_path, "simpler_basic.cfg")

def game_init():
  print("[+] Initializing Doom ...")
  game = vzd.DoomGame()
  game.load_config(config_file_path)
  game.set_window_visible(True)
  game.set_mode(vzd.Mode.PLAYER)
  game.set_screen_format(vzd.ScreenFormat.GRAY8)
  game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
  game.init()
  print("[+] Game Initialized")
  return game

if __name__ == '__main__':
  game = game_init()

