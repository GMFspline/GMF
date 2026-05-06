
import numpy as np
from geomdl import BSpline
import torch

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import random


def shcurve(batch_idx: int):

    casef = 5

    inp = torch.load(r'oripo.pth', map_location='cpu')

    inp = inp.reshape(-1, 3).cpu().detach().numpy().tolist()
    curve = BSpline.Curve()
    curve.degree = 3

    fd = torch.load(r'newpo.pth', map_location='cpu').reshape(-1)
    f = fd[-36:].reshape(12, 3)
    new_po = f.cpu().detach().numpy()

    po = new_po.tolist()

    noise_magnitude = 0.08
    num_points_to_noise = 6
    for _ in range(num_points_to_noise):
        r, c = random.randint(0, 11), random.randint(0, 2)
        po[r][c] += random.uniform(-noise_magnitude, noise_magnitude)

    curve.ctrlpts = po
    knotu = [0, 0, 0, 0]
    step = 1.0 / (len(po) - 3)
    for i in range(len(po) - 4):
        knotu.append((i + 1) * step)
    for i in range(4):
        knotu.append(1.0)

    knot = torch.load('knot.pth', map_location='cpu').reshape(-1)[:16].cpu().detach().numpy().tolist()

    if casef == 1:
        knot = knotu

    curve.knotvector = knotu

    nzinput = []
    nzinput.append(inp[0])
    for i in range(1, len(inp)):
        if inp[i - 1][0] == inp[i][0] and inp[i - 1][1] == inp[i][1] and inp[i - 1][2] == inp[i][2]:
            nzinput.pop()
            break
        nzinput.append(inp[i])
    step = 0.01
    step_num = int((1 / step) + 1)
    oldx = np.zeros(step_num, dtype=float)
    oldy = np.zeros(step_num, dtype=float)
    oldz = np.zeros(step_num, dtype=float)

    for uid in range(step_num):
        pu = uid * step
        oldpa = curve.evaluate_single(pu)
        oldx[uid] = oldpa[0]
        oldy[uid] = oldpa[1]
        oldz[uid] = oldpa[2]

    nzinput = np.array(nzinput)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.scatter(nzinput[:, 0], nzinput[:, 1], nzinput[:, 2], color='red')
    ax1.plot(oldx, oldy, oldz, color='blue')

    plt.show()
    plt.close()
