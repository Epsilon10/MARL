import torch
import numpy as np
from mlagents_envs.registry import default_registry

from unityml_trainer import UnityMLTrainer

unityml_trainer = UnityMLTrainer(env=default_registry["GridWorld"].make())

unityml_trainer.fill_replay_buffer()

NUM_TRAINING_STEPS = 70
# The number of experiences to collect per training step
NUM_NEW_EXP = 1000
# The maximum size of the Buffer
BUFFER_SIZE = 10000



unityml_trainer.env.close()