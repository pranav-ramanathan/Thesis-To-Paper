"""Project DQN reference; GUI/Gym metadata omitted, learning unchanged.

Derived from the preserved reference. See provenance.json.
"""
from types import SimpleNamespace
import random
import numpy as np
import torch
from torch import nn, optim

class HPProteinFoldingEnv:
    """
    Coordinate-based environment with 3D rendering capability.
    Includes trap detection, symmetry-breaking constraints, and
    a bounded coordinate system limiting positions to within +/- (length/2).
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, sequence="PPHHHPHHHHHHHHPPPHHHHHHHHHHPHPPPHHHHHHHHHHHHPPPPHHHHHHPHHPHP"):
        super(HPProteinFoldingEnv, self).__init__()
        self.sequence = sequence
        self.length = len(sequence)
        self.action_space = SimpleNamespace(n=5)

        # Define bounding region based on sequence length
        self.radius = self.length // 2

        # Define observation space
        # Flags to handle symmetry-breaking
        self.has_non_forward_turn = False
        self.has_z_deviation = False

        self.reset()

    def reset(self):
        self.positions = [(0, 0, 0), (1, 0, 0)]
        self.current_index = 2
        self.done = False
        self.has_non_forward_turn = False
        self.has_z_deviation = False
        return self._get_observation()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0.0, True, {}

        if self.current_index >= self.length:
            # All placed
            reward = self._calculate_hh_bonds()
            self.done = True
            return self._get_observation(), reward, True, {}

        next_pos = self._get_next_position(action)
        # Check if next_pos is out of bound
        if not self._in_bounds(next_pos):
            # Treat out-of-bounds as invalid
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, {}
            else:
                return self._get_observation(), 0.0, False, {}

        if next_pos in self.positions:
            # Invalid action, no move
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, {}
            else:
                return self._get_observation(), 0.0, False, {}

        # Valid move
        self.positions.append(next_pos)
        self.current_index += 1

        # Update the symmetry-breaking flags after a valid move
        if not self.has_non_forward_turn and action != 0:
            # The first non-forward action must be to the right (action=4)
            if action == 4:
                self.has_non_forward_turn = True

        if not self.has_z_deviation and action in {1, 2}:
            # The first vertical deviation must be up (action=1).
            if action == 1:
                self.has_z_deviation = True

        if self.current_index == self.length:
            # Finished
            reward = self._calculate_hh_bonds()
            self.done = True
            return self._get_observation(), reward, True, {}
        else:
            if self._is_trapped():
                self.done = True
                return self._get_observation(), 0.0, True, {}
            else:
                return self._get_observation(), 0.0, False, {}

    def _get_observation(self):
        obs = np.zeros((self.length, 5), dtype=np.float32)
        denom = (self.length - 1) if self.length > 1 else 1
        for i, pos in enumerate(self.positions):
            x, y, z = pos
            aatype = 1 if self.sequence[i] == 'H' else 0
            index_norm = i / denom
            obs[i] = [x, y, z, aatype, index_norm]
        for i in range(self.current_index, self.length):
            obs[i] = [0, 0, 0, -1, -1]
        return obs.flatten()

    def _get_direction_vector(self):
        if len(self.positions) < 2:
            return (1, 0, 0)
        p1 = self.positions[-2]
        p2 = self.positions[-1]
        return (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])

    def _get_next_position(self, action):
        f = self._get_direction_vector()
        if f == (1, 0, 0) or f == (-1, 0, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (0, 1, 0) if f == (1, 0, 0) else (0, -1, 0)
            right = (0, -1, 0) if f == (1, 0, 0) else (0, 1, 0)
        elif f == (0, 1, 0) or f == (0, -1, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (1, 0, 0) if f == (0, 1, 0) else (-1, 0, 0)
            right = (-1, 0, 0) if f == (0, 1, 0) else (1, 0, 0)
        else:
            # f=(0,0,1) or f=(0,0,-1)
            up = (0, 1, 0) if f == (0, 0, 1) else (0, -1, 0)
            down = (0, -1, 0) if f == (0, 0, 1) else (0, 1, 0)
            left = (1, 0, 0)
            right = (-1, 0, 0)

        current_pos = self.positions[-1]
        if action == 0:
            move = f
        elif action == 1:
            move = up
        elif action == 2:
            move = down
        elif action == 3:
            move = left
        else:
            move = right
        return (current_pos[0] + move[0], current_pos[1] + move[1], current_pos[2] + move[2])

    def _calculate_hh_bonds(self):
        coords_h = []
        for i, pos in enumerate(self.positions):
            if self.sequence[i] == 'H':
                coords_h.append((i, pos))
        count = 0
        n = len(coords_h)
        for i in range(n):
            for j in range(i + 1, n):
                idx_i, p_i = coords_h[i]
                idx_j, p_j = coords_h[j]
                if abs(idx_i - idx_j) > 1:
                    if self._are_adjacent(p_i, p_j):
                        count += 1
        return count

    def _are_adjacent(self, p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        dz = abs(p1[2] - p2[2])
        return (dx + dy + dz) == 1

    def _in_bounds(self, pos):
        x, y, z = pos
        return (abs(x) <= self.radius and abs(y) <= self.radius and abs(z) <= self.radius)

    def _is_trapped(self):
        if self.current_index < self.length:
            for a in range(self.action_space.n):
                next_pos = self._get_next_position(a)
                # Check out-of-bounds
                if not self._in_bounds(next_pos):
                    continue
                if next_pos not in self.positions:
                    return False
            return True
        return False
    
    def can_finish_dfs(self, positions, current_index):
        """
        DFS to check if we can eventually place all residues from the current state.
        Also skip any action leading out of the bounding region.
        """
        if current_index >= self.length:
            return True

        all_invalid = True
        for a in range(self.action_space.n):
            next_pos = self._get_next_position_dfs(positions, a)
            # Check out-of-bounds
            if not self._in_bounds(next_pos):
                continue
            if next_pos not in positions:
                all_invalid = False
                new_positions = positions + [next_pos]
                if self.can_finish_dfs(new_positions, current_index+1):
                    return True
        if all_invalid:
            return False
        return True

    def _get_next_position_dfs(self, positions, action):
        if len(positions) < 2:
            f = (1,0,0)
        else:
            f = (positions[-1][0]-positions[-2][0],
                 positions[-1][1]-positions[-2][1],
                 positions[-1][2]-positions[-2][2])

        if f == (1,0,0) or f == (-1,0,0):
            up = (0,0,1)
            down = (0,0,-1)
            left = (0,1,0) if f==(1,0,0) else (0,-1,0)
            right = (0,-1,0) if f==(1,0,0) else (0,1,0)
        elif f == (0,1,0) or f == (0,-1,0):
            up = (0,0,1)
            down = (0,0,-1)
            left = (1,0,0) if f==(0,1,0) else (-1,0,0)
            right = (-1,0,0) if f==(0,1,0) else (1,0,0)
        else:
            up = (0,1,0) if f==(0,0,1) else (0,-1,0)
            down = (0,-1,0) if f==(0,0,1) else (0,1,0)
            left = (1,0,0)
            right = (-1,0,0)

        current_pos = positions[-1]
        if action == 0:
            move = f
        elif action == 1:
            move = up
        elif action == 2:
            move = down
        elif action == 3:
            move = left
        else:
            move = right
        return (current_pos[0]+move[0], current_pos[1]+move[1], current_pos[2]+move[2])

    def get_valid_actions_dfs(self):
        valid_mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            # Symmetry-breaking constraints
            if not self.has_non_forward_turn and a != 0 and a != 4:
                continue
            if not self.has_z_deviation and a in {1,2} and a != 1:
                continue

            next_pos = self._get_next_position(a)
            # Check bounding region
            if not self._in_bounds(next_pos):
                continue
            if next_pos in self.positions:
                continue
            new_positions = self.positions + [next_pos]
            current_idx = self.current_index + 1
            if self.can_finish_dfs(new_positions, current_idx):
                valid_mask[a] = True

        return valid_mask


    def get_valid_actions(self):
        """
        Simple version without dfs checking.
        This is a simplified version without DFS, which can save some computational complexity,
        But can still prevent most of the trapping senarios by checking one more step.
        """
        valid_mask = np.zeros(self.action_space.n, dtype=bool)
        for a in range(self.action_space.n):
            # Symmetry-breaking constraints
            if not self.has_non_forward_turn and a != 0 and a != 4:
                continue
            if not self.has_z_deviation and a in {1, 2} and a != 1:
                continue

            next_pos = self._get_next_position(a)
            # Check bounding region
            if not self._in_bounds(next_pos):
                continue
            if next_pos in self.positions:
                continue

            # Check if the agent is not trapped after taking this action
            # Simplified: Do not perform DFS, just ensure that at least one valid move remains after this action
            temp_positions = self.positions + [next_pos]
            temp_current_index = self.current_index + 1
            if not self._is_trapped_after_action(temp_positions, temp_current_index):
                valid_mask[a] = True

        return valid_mask

    def _is_trapped_after_action(self, positions, current_index):
        """
        Check if after taking an action, the agent is not immediately trapped.
        """
        if current_index >= self.length:
            return False  # No further actions needed

        for a in range(self.action_space.n):
            next_pos = self._get_next_position_after_action(positions, a)
            if not self._in_bounds(next_pos):
                continue
            if next_pos in positions:
                continue
            # If at least one valid move exists, not trapped
            return False
        return True

    def _get_next_position_after_action(self, positions, action):

        if len(positions) < 2:
            f = (1, 0, 0)
        else:
            f = (positions[-1][0] - positions[-2][0],
                 positions[-1][1] - positions[-2][1],
                 positions[-1][2] - positions[-2][2])

        if f == (1, 0, 0) or f == (-1, 0, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (0, 1, 0) if f == (1, 0, 0) else (0, -1, 0)
            right = (0, -1, 0) if f == (1, 0, 0) else (0, 1, 0)
        elif f == (0, 1, 0) or f == (0, -1, 0):
            up = (0, 0, 1)
            down = (0, 0, -1)
            left = (1, 0, 0) if f == (0, 1, 0) else (-1, 0, 0)
            right = (-1, 0, 0) if f == (0, 1, 0) else (1, 0, 0)
        else:
            # f=(0,0,1) or f=(0,0,-1)
            up = (0, 1, 0) if f == (0, 0, 1) else (0, -1, 0)
            down = (0, -1, 0) if f == (0, 0, 1) else (0, 1, 0)
            left = (1, 0, 0)
            right = (-1, 0, 0)

        current_pos = positions[-1]
        if action == 0:
            move = f
        elif action == 1:
            move = up
        elif action == 2:
            move = down
        elif action == 3:
            move = left
        else:
            move = right
        return (current_pos[0] + move[0], current_pos[1] + move[1], current_pos[2] + move[2])

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.eps = 1e-6  # small epsilon for priority
        self.beta = 0.4   # initial beta for importance sampling
        self.beta_increment_per_sampling = 1e-6

    def push(self, s, a, r, ns, d):
        max_prio = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((s,a,r,ns,d))
        else:
            self.memory[self.pos] = (s,a,r,ns,d)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos+1)%self.capacity

    def sample(self, batch_size):
        # Compute sampling probabilities
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.memory)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[i] for i in indices]

        # Compute importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices])**(-self.beta)
        weights /= weights.max()

        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.FloatTensor(np.array(next_states)),
                torch.BoolTensor(np.array(dones)),
                torch.FloatTensor(weights),
                indices)

    def update_priorities(self, batch_indices, batch_priorities):
        # Update priorities based on TD errors (batch_priorities = abs(TD_error) + eps)
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + self.eps

    def __len__(self):
        return len(self.memory)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-np.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # odd d_model
            pe[:,1::2] = torch.cos(position * div_term[:(d_model//2)+1])
        else:
            pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        S = x.size(1)
        x = x + self.pe[:, :S, :]
        return x

class DuelingTransformerQNetwork(nn.Module):
    def __init__(self, length, num_actions=5, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(DuelingTransformerQNetwork, self).__init__()
        self.length = length
        self.num_actions = num_actions
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward

        coords_proj_dim = d_model // 2
        type_emb_dim = d_model // 4
        index_proj_dim = d_model - coords_proj_dim - type_emb_dim

        self.coord_proj = nn.Linear(3, coords_proj_dim)
        self.type_emb = nn.Embedding(3, type_emb_dim)
        self.index_proj = nn.Linear(1, index_proj_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.value_fc = nn.Linear(d_model, 1)           # Outputs a single value for the state
        self.advantage_fc = nn.Linear(d_model, num_actions)  # Outputs advantage for each action

    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, self.length, 5)
        coords = seq[:, :, :3]
        aatype = seq[:, :, 3].long()
        index_norm = seq[:, :, 4].unsqueeze(-1)

        aatype = torch.where(aatype == -1, torch.tensor(2, device=aatype.device), aatype)

        coord_emb = self.coord_proj(coords)
        type_emb = self.type_emb(aatype)
        index_emb = self.index_proj(index_norm)

        emb = torch.cat([coord_emb, type_emb, index_emb], dim=2)  # (B, L, d_model) concatenate

        cls_tokens = self.cls_token.expand(B, -1, -1)
        emb = torch.cat([cls_tokens, emb], dim=1)
        emb = self.pos_encoder(emb)
        emb = emb.transpose(0,1)

        encoded = self.transformer_encoder(emb)
        cls_rep = encoded[0]  # (B, d_model)

        # Compute value and advantage
        value = self.value_fc(cls_rep)            # (B, 1)
        advantage = self.advantage_fc(cls_rep)    # (B, num_actions)

        # Q = V + A - mean(A)
        advantage_mean = advantage.mean(dim=1, keepdim=True)  # (B,1)
        q_values = value + advantage - advantage_mean
        return q_values

class DQNAgent:
    def __init__(self, env, length, num_actions=5, gamma=0.99, lr=1e-3, batch_size=64,
                 memory_size=50000, target_update_freq=1000, device='cpu',
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        self.env = env
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.policy_net = DuelingTransformerQNetwork(length, num_actions, d_model, nhead, num_layers, dim_feedforward).to(device)
        self.target_net = DuelingTransformerQNetwork(length, num_actions, d_model, nhead, num_layers, dim_feedforward).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayMemory(memory_size)
        self.steps_done = 0
        self.target_update_freq = target_update_freq

    def select_action(self, state, epsilon):
        valid_actions = self.env.get_valid_actions()  # Boolean mask of size num_actions
        valid_indices = np.where(valid_actions)[0]    # Indices of valid actions

        if random.random() < epsilon:
            # Random action from valid actions only
            if len(valid_indices) > 0:
                return np.random.choice(valid_indices)
            else:
                # If no valid actions (very rare), just pick random action
                return random.randrange(self.num_actions)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t).cpu().numpy().squeeze()

            # Set invalid actions to a large negative number so they won't be chosen
            q_values[~valid_actions] = -1e9
            return q_values.argmax()

    def store_transition(self, s,a,r,ns,d):
        self.memory.push(s,a,r,ns,d)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN target
            next_q_values_policy = self.policy_net(next_states)
            best_actions = next_q_values_policy.argmax(dim=1, keepdim=True)
            next_q_values_target = self.target_net(next_states)
            max_next_q_values = next_q_values_target.gather(1, best_actions).squeeze(1)

            target_values = rewards + (1 - dones.float()) * self.gamma * max_next_q_values

        # Compute TD error
        td_errors = target_values - state_action_values        
        loss = (td_errors.pow(2) * weights).mean() 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
