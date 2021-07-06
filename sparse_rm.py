# store reward machine state in replay buffer

# seperate replay buffer for each reward machine state

# sac for each reward machine state

# each rm state has seperate q function


# reward machine: initial state. intermediate state, final state

from typing import NamedTuple, ByteString, List
import numpy as np

class RMEntry(NamedTuple):
    state: np.ndarray
    next_state: np.ndarray
    event: str
    reward: float

class SparseRewardMachine():
    def __init__(self, rm_entries: List[RMEntry]):
        self.rm_states = []
        self.events = set()
        self.init_state = None
        self.state_transition_func = {}
        self.reward_transition_func = {}
        self.terminal_states = set()
    
    def load_reward_machine(self, rm_entries):
        self.init_state = rm_entries[0].state

        for rm_entry in rm_entries:
            self._add_transition(rm_entry.state, rm_entry.next_state, )
    
    def _is_terminal(self, rm_state):
        for s1 in self.reward_transition_func:
            for s2 in self.reward_transition_func[s1]:
                if self.reward_transition_func[s1][s2] == 1:
                    return True
        return False

    
    def _add_state(self, rm_state_list):
        self.rm_states.extend(
            [state for state in rm_state_list if state not in self.machine_states]
        )
        
    def _add_transition(self, rm_state_1, rm_state_2, event, reward):
        self._add_state([rm_state_1, rm_state_2])

        if rm_state_1 not in self.state_transition_func:
            self.state_transition_func[rm_state_1] = {}
        if event not in self.state_transition_func[rm_state_1]:
            self.state_transition_func[rm_state_1][event] = rm_state_2
        
        if rm_state_1 not in self.reward_transition_func:
            self.reward_transition_func[rm_state_1] = {}
        self.reward_transition_func[rm_state_1][rm_state_2] = reward

    
