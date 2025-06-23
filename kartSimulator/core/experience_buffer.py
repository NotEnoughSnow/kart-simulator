import random
from collections import deque


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_experience(self, obs, action, reward, next_obs, done, source):
        """Store experience tuple in the buffer, with a 'source' flag for agent or expert."""
        self.buffer.append((obs, action, reward, next_obs, done, source))

    def sample(self, batch_size, expert_ratio=0.5):
        """Sample mixed data with a set ratio of expert to agent data."""
        expert_data = [exp for exp in self.buffer if exp[5] == 'expert']
        agent_data = [exp for exp in self.buffer if exp[5] == 'agent']

        n_expert = int(batch_size * expert_ratio)
        n_agent = batch_size - n_expert

        sampled_expert = random.sample(expert_data, min(n_expert, len(expert_data)))
        sampled_agent = random.sample(agent_data, min(n_agent, len(agent_data)))

        return sampled_expert + sampled_agent