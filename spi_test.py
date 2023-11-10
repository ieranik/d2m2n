import sys
import matplotlib.pyplot as plt

import numpy as np
from numpy import load

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F


weights = '/content/s16_k28.pth'
data_path = '/content/spi_16.npz'
imsize = 16
lr = 0.002
k = 20
l_i = 2
l_h = 150
l_q = 10
batch_size = 128
plot = False


def print_env(a, sx, sy, dx, dy, dis):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i][j] == 0:
                if sx[0] == i and sy[0] == j:
                    print("S ", end = '')
                elif dx[0] == i and dy[0] == j:
                    print("D ", end = '')
                else:
                    print(". ", end = '')
            else:  
                print("0 ", end = '')
        print("")
    print(dis[0])

class VIN(nn.Module):
    def __init__(self):
        super(VIN, self).__init__()
        self.h = nn.Conv2d(
            in_channels=l_i,
            out_channels=l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.zeros(l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_view, state_x, state_y, k):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and softmaxed logits
        """
        
        
        h = self.h(input_view)  # Intermediate output
        r = self.r(h)           # Reward
        q = self.q(r)           # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        def eval_q(r, v):
            return F.conv2d(
                # Stack reward with most recent value    
                torch.cat([r, v], 1),
                # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)

        # Update q and v values
        for i in range(k - 1):
            q = eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = eval_q(r, v)
        # q: (batch_sz, l_q, map_size, map_size)
        batch_sz, l_q, _, _ = q.size()
        q_out = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)

        logits = self.fc(q_out)  # q_out to actions

        return logits, self.sm(logits)

def main():
    # Correct vs total:
    correct, total = 0.0, 0.0
    # Instantiate a VIN model
    vin: VIN = VIN()
    # Load model parameters
    vin.load_state_dict(torch.load(weights))
    #, map_location=torch.device('cpu')
    # Automatically select device to make the code device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vin = vin.to(device)

    scnt = 0
    ccnt = 0
    dcnt = 0
    sm = 0


    data = load(data_path)
    ln = len(data['arr_0'])

    for i in range(1000):

        X_in = data['arr_0'][i]
        X_in = X_in.astype(int)
        X_in = torch.from_numpy(X_in)
        X_in = X_in.reshape(1, 2, imsize, imsize)

        #print_env(data['arr_0'][i][0], data['arr_1'][i], data['arr_2'][i], data['arr_3'][i], data['arr_4'][i], data['arr_5'][i])

        plen = 0
        cur_x = data['arr_1'][i]
        cur_y = data['arr_2'][i]
        dx = data['arr_3'][i]
        dy = data['arr_4'][i]
        while True:
            #print(cur_x, cur_y)
            if cur_x == dx and cur_y == dy:
                scnt = scnt + 1
                sm = sm + (data['arr_5'][i][0] / plen)
                break
            else:
                S1_in = cur_x
                S1_in = S1_in.astype(int)
                
                S2_in = cur_y
                S2_in = S2_in.astype(int)

                S1_in = torch.from_numpy(S1_in)
                S2_in = torch.from_numpy(S2_in)

                # Get input batch
                X_in, S1_in, S2_in = [d.float().to(device) for d in [X_in, S1_in, S2_in]]

                # Forward pass in our neural net
                _, predictions = vin(X_in, S1_in, S2_in, k)
                #print(predictions)
                _, indices = torch.max(predictions.cpu(), 1, keepdim=True)
                #print(indices)

                dir = indices[0][0].item()


                #N=(-1, 0), S=(1, 0), E=(0, 1), W=(0, -1), NE=(-1, 1), NW=(-1, -1), SE=(1, 1), SW=(1, -1)

                if dir == 0:
                    cur_x -= 1
                elif dir == 1:
                    cur_x += 1
                elif dir == 2:
                    cur_y += 1
                elif dir == 3:
                    cur_y -= 1
                elif dir == 4:
                    cur_x -= 1
                    cur_y += 1
                elif dir == 5:
                    cur_x -= 1
                    cur_y -= 1
                elif dir == 6:
                    cur_x += 1
                    cur_y += 1
                else:
                    cur_x += 1
                    cur_y -= 1
                
                #print(cur_x, cur_y)

                if data['arr_0'][i][0][cur_x[0]][cur_y[0]] == 1:
                    ccnt = ccnt + 1
                    break

                plen += 1

                if plen > 2.5 * imsize:
                    dcnt = dcnt + 1
                    break
        print(scnt, dcnt, ccnt, sm)
                


  

                
            
        
        # Get inputs as expected by network
        
        
        



        #a = indices.data.numpy()[0][0]
        # Transform prediction to indices
        #s = G.map_ind_to_state(pred_traj[j - 1, 0], pred_traj[j - 1, 1])
        #ns = G.sample_next_state(s, a)
        #nr, nc = G.get_coords(ns)
        #pred_traj[j, 0] = nr
        #pred_traj[j, 1] = nc
        #if nr == goal[0] and nc == goal[1]:
            # We hit goal so fill remaining steps
        #    pred_traj[j + 1:, 0] = nr
        #    pred_traj[j + 1:, 1] = nc
        #    break
        # Plot optimal and predicted path (also start, end)

        #       if pred_traj[-1, 0] == goal[0] and pred_traj[-1, 1] == goal[1]:
        #            correct += 1
        #        total += 1
        #        if config.plot == True:
        #            visualize(G.image.T, states_xy[i], pred_traj)
        #sys.stdout.write("\r" + str(int((float(dom) / n_domains) * 100.0)) + "%")
        #sys.stdout.flush()
    #sys.stdout.write("\n")
    #print('Rollout Accuracy: {:.2f}%'.format(100 * (correct / total)))


def visualize(dom, states_xy, pred_traj):
    fig, ax = plt.subplots()
    implot = plt.imshow(dom, cmap="Greys_r")
    ax.plot(states_xy[:, 0], states_xy[:, 1], c='b', label='Optimal Path')
    ax.plot(
        pred_traj[:, 0], pred_traj[:, 1], '-X', c='r', label='Predicted Path')
    ax.plot(states_xy[0, 0], states_xy[0, 1], '-o', label='Start')
    ax.plot(states_xy[-1, 0], states_xy[-1, 1], '-s', label='Goal')
    legend = ax.legend(loc='upper right', shadow=False)
    for label in legend.get_texts():
        label.set_fontsize('x-small')   # The legend text size
    for label in legend.get_lines():
        label.set_linewidth(0.5)        # The legend line width
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)


if __name__ == '__main__':
    # Parsing training parameters
    main()
