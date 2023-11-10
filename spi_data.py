import sys
import random
import numpy as np
import torch
import torch.utils.data as data

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from collections import OrderedDict
import matplotlib.pyplot as plt

size = 12
dom_size = [12, 12]
n_domains = 250
max_obs = 50
max_obs_size = 2
n_traj = 5
state_batch_size = 1
lowl = 2
lowh = 3
save_path = "spi_{0}".format(dom_size[0])


class obstacles:
    """A class for generating obstacles in a domain"""

    def __init__(self,
                 domsize=None,
                 mask=None,
                 size_max=None,
                 dom=None,
                 obs_types=None,
                 num_types=None):
        self.domsize = domsize or []
        self.mask = mask or []
        self.dom = dom or np.zeros(self.domsize)
        self.dom = self.dom.astype(int)
        self.obs_types = obs_types or ["circ", "rect"]
        self.num_types = num_types or len(self.obs_types)
        self.size_max = size_max or np.max(self.domsize) / 4

    def place_wall(self, hor, sx, sy, len):
        if hor == 1:
            for i in range(len): self.dom[sx, sy + i] = -1
        else:   
            for i in range(len): self.dom[sx + i, sy] = -1
        
    
    def recur_gen_env(self, sx, sy, x, y):
        if x <= lowh and y <= lowh: return
        if x > y or (x == y and np.random.randint(2) == 1):
            if x < 2 * lowl + 1: 
                return
            else:
                xp = np.random.randint(x - 2 * lowl) + lowl
                self.place_wall(1, sx + xp, sy, y)
                self.recur_gen_env(sx, sy, xp, y)
                self.recur_gen_env(sx + xp + 1, sy, x - xp - 1, y)
        else:
            if y < 2 * lowl + 1: 
                return
            else:
                yp = np.random.randint(y - 2 * lowl) + lowl
                self.place_wall(0, sx, sy + yp, x)
                self.recur_gen_env(sx, sy, x, yp)
                self.recur_gen_env(sx, sy + yp + 1, x, y - yp - 1)
        
    

    def gen_env(self):
        for i in range(self.domsize[0]): 
            for j in range(self.domsize[1]): self.dom[i, j] = 0

        self.place_wall(1, 0, 0, self.domsize[1])
        self.place_wall(0, 0, 0, self.domsize[0])
        self.place_wall(1, self.domsize[0] - 1, 0, self.domsize[1])
        self.place_wall(0, 0, self.domsize[1] - 1, self.domsize[0])

        self.recur_gen_env(1, 1, self.domsize[0] - 2, self.domsize[1] - 2)

        if self.dom[self.mask[0], self.mask[1]] == 0: 
            return True
        else:
            return False


    def door_dfs_vis(self, u, visited, L, el):
        visited[u] = 1
        for i in range(len(L[u])):
            v = L[u][i]
            if visited[v] == 0:
                el.append([u, v])
                self.door_dfs_vis(v, visited, L, el)
        return el
  

    def door_dfs(self, L, el):
        sz = len(L)
        visited = np.zeros(sz)
        visited = visited.astype(int)
        return self.door_dfs_vis(np.random.randint(sz), visited, L, el)
    
    
    def place_doors(self):
        room_no = 1
        dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        for i in range(self.domsize[0] - 1):
            for j in range(self.domsize[1] - 1):
                if self.dom[i, j] == 0:
                    flag = False
                    for k in range(4):
                        color = self.dom[i + dir[k][0], j + dir[k][1]]
                        if color > 0:
                            flag = True
                            self.dom[i, j] = color              
                    if flag == False: 
                        self.dom[i, j] = room_no
                        room_no = room_no + 1

        num_room = room_no - 1
        M = np.zeros((num_room, num_room))

        for i in range(self.domsize[0] - 1):
            for j in range(self.domsize[1] - 1):
                if self.dom[i, j] == -1:
                    if self.dom[i + 1, j] != -1 and self.dom[i - 1, j] != -1:    
                        cellno = i * self.domsize[0] + j;
                        if M[self.dom[i + 1, j] - 1][self.dom[i - 1, j] - 1] == 0:     
                            M[self.dom[i + 1, j] - 1][self.dom[i - 1, j] - 1] = cellno
                            M[self.dom[i - 1, j] - 1][self.dom[i + 1, j] - 1] = cellno     
                        else:  
                            if np.random.randint(3) == 0:
                                M[self.dom[i + 1, j] - 1][self.dom[i - 1, j] - 1] = cellno
                                M[self.dom[i - 1, j] - 1][self.dom[i + 1, j] - 1] = cellno
                    elif self.dom[i, j + 1] != -1 and self.dom[i, j - 1] != -1:
                        cellno = i * self.domsize[0] + j
                        if M[self.dom[i, j + 1] - 1][self.dom[i, j + 1] - 1] == 0:
                            M[self.dom[i, j + 1] - 1][self.dom[i, j - 1] - 1] = cellno
                            M[self.dom[i, j - 1] - 1][self.dom[i, j + 1] - 1] = cellno
                        else:
                            if np.random.randint(3) == 0:  
                                M[self.dom[i, j + 1] - 1][self.dom[i, j - 1] - 1] = cellno
                                M[self.dom[i, j - 1] - 1][self.dom[i, j + 1] - 1] = cellno                     

        L = []
        for i in range(num_room):
            tmp = []
            for j in range(num_room):
                if M[i][j] != 0:
                    tmp.append(j)
            random.shuffle(tmp)
            L.append(tmp)

        el = []
        el = self.door_dfs(L, el)

        for i in range(len(el)):
            cellno = M[el[i][0]][el[i][1]]
            cellx = cellno // self.domsize[0]
            celly = cellno % self.domsize[0]
            cellx = cellx.astype(int)
            celly = celly.astype(int)
            self.dom[cellx, celly] = 0
      
        for i in range(self.domsize[0]):
            for j in range(self.domsize[1]):
                if self.dom[i, j] == -1: 
                    self.dom[i, j] = 1
                else:
                    self.dom[i, j] = 0


    def get_final(self):
        # Process obstacle map for domain
        im = np.copy(self.dom)
        im = np.max(im) - im
        im = im / np.max(im)
        return im

    def show(self):
        # Utility function to view obstacle map
        plt.imshow(self.get_final(), cmap='Greys')
        plt.show()

    def _print(self):
        # Utility function to view obstacle map
        #  information
        print("domsize: ", self.domsize)
        print("mask: ", self.mask)
        print("dom: ", self.dom)
        print("obs_types: ", self.obs_types)
        print("num_types: ", self.num_types)
        print("size_max: ", self.size_max)



class GridWorld:
    """A class for making gridworlds"""

    ACTION = OrderedDict(N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1))

    def __init__(self, image, target_x, target_y):
        self.image = image
        self.n_row = image.shape[0]
        self.n_col = image.shape[1]
        self.obstacles = np.where(self.image == 0)
        self.freespace = np.where(self.image != 0)
        self.target_x = target_x
        self.target_y = target_y
        self.n_states = self.n_row * self.n_col
        self.n_actions = len(self.ACTION)

        self.G, self.W, self.P, self.R, self.state_map_row, self.state_map_col = self.set_vals()

    def loc_to_state(self, row, col):
        return np.ravel_multi_index([row, col], (self.n_row, self.n_col), order='F')

    def state_to_loc(self, state):
        return np.unravel_index(state, (self.n_col, self.n_row), order='F')

    def set_vals(self):
        # Setup function to initialize all necessary

        # Cost of each action, equivalent to the length of each vector
        #  i.e. [1., 1., 1., 1., 1.414, 1.414, 1.414, 1.414]
        action_cost = np.linalg.norm(list(self.ACTION.values()), axis=1)
        # Initializing reward function R: (curr_state, action) -> reward: float
        # Each transition has negative reward equivalent to the distance of transition
        R = - np.ones((self.n_states, self.n_actions)) * action_cost
        # Reward at target is zero
        target = self.loc_to_state(self.target_x, self.target_y)
        R[target, :] = 0

        # Transition function P: (curr_state, next_state, action) -> probability: float
        P = np.zeros((self.n_states, self.n_states, self.n_actions))
        # Filling in P
        for row in range(self.n_row):
            for col in range(self.n_col):
                curr_state = self.loc_to_state(row, col)
                for i_action, action in enumerate(self.ACTION):
                    neighbor_row, neighbor_col = self.move(row, col, action)
                    neighbor_state = self.loc_to_state(neighbor_row, neighbor_col)
                    P[curr_state, neighbor_state, i_action] = 1

        # Adjacency matrix of a graph connecting curr_state and next_state
        G = np.logical_or.reduce(P, axis=2)
        # Weight of transition edges, equivalent to the cost of transition
        W = np.maximum.reduce(P * action_cost, axis=2)

        non_obstacles = self.loc_to_state(self.freespace[0], self.freespace[1])

        non_obstacles = np.sort(non_obstacles)

        G = G[non_obstacles, :][:, non_obstacles]
        W = W[non_obstacles, :][:, non_obstacles]
        P = P[non_obstacles, :, :][:, non_obstacles, :]
        R = R[non_obstacles, :]

        state_map_col, state_map_row = np.meshgrid(
            np.arange(0, self.n_col), np.arange(0, self.n_row))
        state_map_row = state_map_row.flatten('F')[non_obstacles]
        state_map_col = state_map_col.flatten('F')[non_obstacles]

        return G, W, P, R, state_map_row, state_map_col

    def get_graph(self):
        # Returns graph
        G = self.G
        W = self.W[self.W != 0]
        return G, W

    def get_graph_inv(self):
        # Returns transpose of graph
        G = self.G.T
        W = self.W.T
        return G, W

    def val_2_image(self, val):
        # Zeros for obstacles, val for free space
        im = np.zeros((self.n_row, self.n_col))
        im[self.freespace[0], self.freespace[1]] = val
        return im

    def get_value_prior(self):
        # Returns value prior for gridworld
        s_map_col, s_map_row = np.meshgrid(
            np.arange(0, self.n_col), np.arange(0, self.n_row))
        im = np.sqrt(
            np.square(s_map_col - self.target_y) +
            np.square(s_map_row - self.target_x))
        return im

    def get_reward_prior(self):
        # Returns reward prior for gridworld
        im = -1 * np.ones((self.n_row, self.n_col))
        im[self.target_x, self.target_y] = 10
        return im

    def t_get_reward_prior(self):
        # Returns reward prior as needed for
        #  dataset generation
        im = np.zeros((self.n_row, self.n_col))
        im[self.target_x, self.target_y] = 10
        return im

    def get_state_image(self, row, col):
        # Zeros everywhere except [row,col]
        im = np.zeros((self.n_row, self.n_col))
        im[row, col] = 1
        return im

    def map_ind_to_state(self, row, col):
        # Takes [row, col] and maps to a state
        rw = np.where(self.state_map_row == row)
        cl = np.where(self.state_map_col == col)
        return np.intersect1d(rw, cl)[0]

    def get_coords(self, states):
        # Given a state or states, returns
        #  [row,col] pairs for the state(s)
        non_obstacles = self.loc_to_state(self.freespace[0], self.freespace[1])
        non_obstacles = np.sort(non_obstacles)
        states = states.astype(int)
        r, c = self.state_to_loc(non_obstacles[states])
        return r, c

    def rand_choose(self, in_vec):
        # Samples
        if len(in_vec.shape) > 1:
            if in_vec.shape[1] == 1:
                in_vec = in_vec.T
        temp = np.hstack((np.zeros((1)), np.cumsum(in_vec))).astype('int')
        q = np.random.rand()
        x = np.where(q > temp[0:-1])
        y = np.where(q < temp[1:])
        return np.intersect1d(x, y)[0]

    def next_state_prob(self, s, a):
        # Gets next state probability for
        #  a given action (a)
        if hasattr(a, "__iter__"):
            p = np.squeeze(self.P[s, :, a])
        else:
            p = np.squeeze(self.P[s, :, a]).T
        return p

    def sample_next_state(self, s, a):
        # Gets the next state given the
        #  current state (s) and an
        #  action (a)
        vec = self.next_state_prob(s, a)
        result = self.rand_choose(vec)
        return result

    def get_size(self):
        # Returns domain size
        return self.n_row, self.n_col

    def move(self, row, col, action):
        # Returns new [row,col]
        #  if we take the action
        r_move, c_move = self.ACTION[action]
        new_row = max(0, min(row + r_move, self.n_row - 1))
        new_col = max(0, min(col + c_move, self.n_col - 1))
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


class GridworldData(data.Dataset):
    def __init__(self,
                 file,
                 imsize,
                 train=True,
                 transform=None,
                 target_transform=None):
        assert file.endswith('.npz')  # Must be .npz format
        self.file = file
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or test set

        self.images, self.S1, self.S2, self.labels =  \
                                self._process(file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        label = self.labels[index]
        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        else:  # Internal default transform: Just to Tensor
            img = torch.from_numpy(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(s1), int(s2), int(label)

    def __len__(self):
        return self.images.shape[0]

    def _process(self, file, train):
        """Data format: A list, [train data, test data]
        Each data sample: label, S1, S2, Images, in this order.
        """
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
                S1 = f['arr_1']
                S2 = f['arr_2']
                labels = f['arr_3']
            else:
                images = f['arr_4']
                S1 = f['arr_5']
                S2 = f['arr_6']
                labels = f['arr_7']
        # Set proper datatypes
        images = images.astype(np.float32)
        S1 = S1.astype(int)  # (S1, S2) location are integers
        S2 = S2.astype(int)
        labels = labels.astype(int)  # Labels are integers
        # Print number of samples
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, S1, S2, labels



def extract_action(traj):
    # Given a trajectory, outputs a 1D vector of
    #  actions corresponding to the trajectory.
    n_actions = 8
    action_vecs = np.asarray([[-1., 0.], [1., 0.], [0., 1.], [0., -1.],
                              [-1., 1.], [-1., -1.], [1., 1.], [1., -1.]])
    action_vecs[4:] = 1 / np.sqrt(2) * action_vecs[4:]
    action_vecs = action_vecs.T
    state_diff = np.diff(traj, axis=0)
    norm_state_diff = state_diff * np.tile(
        1 / np.sqrt(np.sum(np.square(state_diff), axis=1)), (2, 1)).T
    prj_state_diff = np.dot(norm_state_diff, action_vecs)
    actions_one_hot = np.abs(prj_state_diff - 1) < 0.00001
    actions = np.dot(actions_one_hot, np.arange(n_actions).T)
    return actions


def trace_path(pred, source, target):
    # traces back shortest path from
    #  source to target given pred
    #  (a predicessor list)
    max_len = 1000
    path = np.zeros((max_len, 1))
    i = max_len - 1
    path[i] = target
    while path[i] != source and i > 0:
        try:
            path[i - 1] = pred[int(path[i])]
            i -= 1
        except Exception as e:
            return []
    if i >= 0:
        path = path[i:]
    else:
        path = None
    return path


def sample_trajectory(M: GridWorld, n_states):
    # Samples trajectories from random nodes
    #  in our domain (M)
    G, W = M.get_graph_inv()
    N = G.shape[0]
    if N >= n_states:
        rand_ind = np.random.permutation(N)
    else:
        rand_ind = np.tile(np.random.permutation(N), (1, 10))
    init_states = rand_ind[0:n_states].flatten()
    goal_s = M.map_ind_to_state(M.target_x, M.target_y)
    states = []
    states_xy = []
    states_one_hot = []
    # Get optimal path from graph
    g_dense = W
    g_masked = np.ma.masked_values(g_dense, 0)
    g_sparse = csr_matrix(g_dense)
    d, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)
    for i in range(n_states):
        path = trace_path(pred, goal_s, init_states[i])
        path = np.flip(path, 0)
        states.append(path)
    for state in states:
        L = len(state)
        r, c = M.get_coords(state)
        row_m = np.zeros((L, M.n_row))
        col_m = np.zeros((L, M.n_col))
        for i in range(L):
            row_m[i, r[i]] = 1
            col_m[i, c[i]] = 1
        states_one_hot.append(np.hstack((row_m, col_m)))
        states_xy.append(np.hstack((r, c)))
    return states_xy, states_one_hot


def make_data(dom_size, n_domains, max_obs, max_obs_size, n_traj,
              state_batch_size):

    X_l = []
    S1_l = []
    S2_l = []
    D1 = []
    D2 = []
    Labels_l = []
    Ln = []

    dom = 0.0
    while dom <= n_domains:
        goal = [1 + np.random.randint(dom_size[0] - 2), 1 + np.random.randint(dom_size[1] - 2)]
        # Generate obstacle map
        obs = obstacles([dom_size[0], dom_size[1]], goal, max_obs_size)
        # Add obstacles to map
        # n_obs = obs.add_n_rand_obs(max_obs)
        # # Add border to map
        # border_res = obs.add_border()
        # # Ensure we have valid map
        # if n_obs == 0 or not border_res:
        #     continue

        while True:
          flag = obs.gen_env()
          if flag == True:    
              break
        obs.place_doors()

        # Get final map
        im = obs.get_final()
        # Generate gridworld from obstacle map
        G = GridWorld(im, goal[0], goal[1])
        # Get value prior
        value_prior = G.t_get_reward_prior()
        # Sample random trajectories to our goal
        states_xy, states_one_hot = sample_trajectory(G, n_traj)
        for i in range(n_traj):
            if len(states_xy[i]) > 1:
                # Get optimal actions for each state
                actions = extract_action(states_xy[i])
                ln = states_xy[i].shape[0] - 1
                ns = 1
                # Invert domain image => 0 = free, 1 = obstacle
                image = 1 - im
                # Resize domain and goal images and concate
                image_data = np.resize(image, (1, 1, dom_size[0], dom_size[1]))
                value_data = np.resize(value_prior,
                                       (1, 1, dom_size[0], dom_size[1]))
                iv_mixed = np.concatenate((image_data, value_data), axis=1)
                X_current = np.tile(iv_mixed, (ns, 1, 1, 1))
                # Resize states
                S1_current = np.expand_dims(states_xy[i][0:ns, 0], axis=1)
                S2_current = np.expand_dims(states_xy[i][0:ns, 1], axis=1)

                D1_current = np.ndarray([1,1])
                D1_current[0, 0] = goal[0]
                D1_current = D1_current.astype(int)

                D2_current = np.ndarray([1,1])
                D2_current[0, 0] = goal[1]
                D2_current = D2_current.astype(int)
                
                # Resize labels
                

                Ln_current = np.ndarray([1,1])
                Ln_current[0, 0] = ln
                Ln_current = Ln_current.astype(int)
                # Append to output list
                X_l.append(X_current)
                S1_l.append(S1_current)
                S2_l.append(S2_current)
                D1.append(D1_current)
                D2.append(D2_current)
                Ln.append(Ln_current)
        dom += 1
        sys.stdout.write("\r" + str(int((dom / n_domains) * 100)) + "%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    # Concat all outputs
    X_f = np.concatenate(X_l)
    S1_f = np.concatenate(S1_l)
    S2_f = np.concatenate(S2_l)
    D1_f = np.concatenate(D1)
    D2_f = np.concatenate(D2)
    Ln_f = np.concatenate(Ln)
    return X_f, S1_f, S2_f, D1_f, D2_f, Ln_f


def main(dom_size=(28, 28),
         n_domains=100,
         max_obs=50,
         max_obs_size=2,
         n_traj=7,
         state_batch_size=1):
    # Get path to save dataset
    
    # Get training data
    print("Now making training data...")
    X_out, S1_out, S2_out, D1_out, D2_out, Ln_out = make_data(
        dom_size, n_domains, max_obs, max_obs_size, n_traj, state_batch_size)
    # Get testing data
    #print("\nNow making  testing data...")
    #X_out_ts, S1_out_ts, S2_out_ts, Labels_out_ts = make_data(
    #    dom_size, n_domains / 6, max_obs, max_obs_size, n_traj,
    #    state_batch_size)
    # Save dataset
    np.savez_compressed(save_path, X_out, S1_out, S2_out, D1_out, D2_out, Ln_out)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--size", "-s", type=int, help="size of the domain", default=28)
    # parser.add_argument("--n_domains", "-nd", type=int, help="number of domains", default=5000)
    # parser.add_argument("--max_obs", "-no", type=int, help="maximum number of obstacles", default=50)
    # parser.add_argument("--max_obs_size", "-os", type=int, help="maximum obstacle size", default=2)
    # parser.add_argument("--n_traj", "-nt", type=int, help="number of trajectories", default=7)
    # parser.add_argument("--state_batch_size", "-bs", type=int, help="state batch size", default=1)

    # args = parser.parse_args()
    # size = args.size

    main(dom_size=(size, size), n_domains=n_domains, max_obs=max_obs,
         max_obs_size=max_obs_size, n_traj=n_traj, state_batch_size=state_batch_size)
