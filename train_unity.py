import torch
import numpy as np
from mlagents_envs.registry import default_registry

from unityml_trainer import UnityMLTrainer

unityml_trainer = UnityMLTrainer(env=default_registry["GridWorld"].make())

unityml_trainer.run()
unityml_trainer.close()