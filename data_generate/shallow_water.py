#shallow water propagation
"""
Solution of Shallow-water equations using a Python class.
Adapted for Python training course at CNRS from https://github.com/mrocklin/ShallowWater/

Dmitry Khvorostyanov, 2015
CNRS/LMD/IPSL, dmitry.khvorostyanov @ lmd.polytechnique.fr
"""

import multiprocessing
from multiprocessing import Pool
import os
import yaml
import numpy as np
# import pandas as pd
# import tqdm


def x_to_y(X):  # averaging in 2*2 windows (4 pixels)
    dim = X.shape[0]
    dim = 20
    Y = np.zeros((int(dim / 2), int(dim / 2)))
    for i in range(int(dim / 2)):
        for j in range(int(dim / 2)):
            Y[i, j] = X[2 * i, 2 * j] + X[2 * i + 1, 2 * j] + X[2 * i, 2 * j + 1] + X[2 * i + 1, 2 * j + 1]
            Y_noise = np.random.multivariate_normal(np.zeros(100), 0.0000 * np.eye(100))
            Y_noise.shape = (10, 10)
            Y = Y + Y_noise
    return Y


class shallow(object):

    # domain

    #N = 100
    #L = 1.
    #dx =  L / N
    #dt = dx / 100.

    # Initial Conditions

    #u = zeros((N,N)) # velocity in x direction
    #v = zeros((N,N)) # velocity in y direction

    #h_ini = 1.
    #h = h_ini * ones((N,N)) # pressure deviation (like height)
    #x,y = mgrid[:N,:N]

    time = 0

    plt = []
    fig = []

    def __init__(
            self,
            x=[],
            y=[],
            h_ini=1.,
            u=[],
            v=[],
            # distance (m), time interval (s)
            dx=0.01,
            dt=0.0001,
            N=64,
            L=1.,
            px=16,
            py=16,
            # Square of radius, height of water colunm
            R=64,
            Hp=0.1,
            g=1.,
            b=0.):  # How define no default argument before?

        # add a perturbation in pressure surface

        self.px, self.py = px, py
        self.R = R
        self.Hp = Hp

        # Physical parameters

        self.g = g
        self.b = b
        self.L = L
        self.N = N

        # limits for h,u,v

        self.dx = dx
        self.dt = dt

        self.x, self.y = np.mgrid[:self.N, :self.N]

        self.u = np.zeros((self.N, self.N))
        self.v = np.zeros((self.N, self.N))

        self.h_ini = h_ini

        self.h = self.h_ini * np.ones((self.N, self.N))

        rr = (self.x - px)**2 + (self.y - py)**2
        self.h[rr < R] = self.h_ini + Hp  #set initial conditions

        self.lims = [(self.h_ini - self.Hp, self.h_ini + self.Hp),
                     (-0.02, 0.02), (-0.02, 0.02)]

    def dxy(self, A, axis=0):
        """
        Compute derivative of array A using balanced finite differences
        Axis specifies direction of spatial derivative (d/dx or d/dy)
        dA[i]/dx =  (A[i+1] - A[i-1] )  / 2dx
        """
        return (np.roll(A, -1, axis) - np.roll(A, 1, axis)) / (
            self.dx * 2.
        )  # roll: shift the array axis=0 shift the horizontal axis

    def d_dx(self, A):
        return self.dxy(A, 1)

    def d_dy(self, A):
        return self.dxy(A, 0)

    def d_dt(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        """
        for x in [h, u, v]:  # type check
            assert isinstance(x, np.ndarray) and not isinstance(x, np.matrix)

        g, b, dx = self.g, self.b, self.dx

        du_dt = -g * self.d_dx(h) - b * u
        dv_dt = -g * self.d_dy(h) - b * v

        H = 0  #h.mean() - our definition of h includes this term
        dh_dt = -self.d_dx(u * (H + h)) - self.d_dy(v * (H + h))

        return dh_dt, du_dt, dv_dt

    def d_dt_conservative(self, h, u, v):
        """
        http://en.wikipedia.org/wiki/Shallow_water_equations#Conservative_form
        """
        for x in [h, u, v]:  # type check
            assert isinstance(x, np.ndarray) and not isinstance(x, np.matrix)

        g, b, dx = self.g, self.b, self.dx

        dh_dt = -self.d_dx(h * u) - self.d_dy(h * v)
        du_dt = (dh_dt * u - self.d_dx(h * u ** 2 + 1. / 2 * g * h ** 2) - self.d_dy(h * u * v)) / h
        dv_dt = (dh_dt * v - self.d_dx(h * u * v) - self.d_dy(h * v ** 2 + 1. / 2 * g * h ** 2)) / h

        return dh_dt, du_dt, dv_dt

    def evolve(self):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        # dh_dt, du_dt, dv_dt = self.d_dt(self.h, self.u, self.v)
        dh_dt, du_dt, dv_dt = self.d_dt_conservative(self.h, self.u, self.v)
        dt = self.dt

        self.h += dh_dt * dt
        self.u += du_dt * dt
        self.v += dv_dt * dt
        self.time += dt

        return self.h, self.u, self.v


def simulation(fig_size, R, Hp_hat, iteration_times):
    SW = shallow(N=fig_size, px=72, py=80, R=R * R, Hp=Hp_hat / 100, b=0.2)

    # chose a point (x,y) to check the evolution
    x, y = 10, 10

    # index = 0

    #SW.plot()
    u_vect = np.zeros(iteration_times)
    v_vect = np.zeros(iteration_times)
    h_vect = np.zeros(iteration_times)

    data_list = []
    for i in range(iteration_times):
        SW.evolve()
        u_vect[i] = SW.u[x][y]
        v_vect[i] = SW.v[x][y]
        h_vect[i] = SW.h[x][y]
        #SW.animate()

        if i % 25 == 0:
            U = np.expand_dims(SW.u, axis=0)  # convert (128,128) to (1,128,128)
            V = np.expand_dims(SW.v, axis=0)  # convert (128,128) to (1,128,128)
            H = np.expand_dims(SW.h, axis=0)  # convert (128,128) to (1,128,128)

            data = [U, V, H]  # aggregate 3 np arrays to one list

            data = np.concatenate(data, axis=0)  # concatenate 3 np arrays by axis 0, here the shape is (3,128,128)

            data = np.expand_dims(data, axis=0)  # convert (3,128,128) to (1,3,128,128)

            data_list.append(data)  # append current data into a list

    final_npy = np.concatenate(data_list, axis=0)  # concatenate all array in this list, there shape is (t,3,128,128)

    timestep = int(iteration_times / 25)
    file_name = f'R_{R}_Hp_{Hp_hat}'
    save_dir = f'../data_v2/test/raw/'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, file_name), final_npy)

    return {
        'file_name': file_name,
        'save_path': os.path.join(save_dir, f"{file_name}.npy")
    }


def worker_simulation(params):
    fig_size, R, Hp, iteration_times = params
    return simulation(fig_size, R, Hp, iteration_times)


if __name__ == "__main__":

    config = yaml.load(open("problem_config.yaml", "r"),
                       Loader=yaml.FullLoader)

    for timestep in config['timestep']:
        # Prepare parameters for all simulations
        timestep = int(timestep)
        fig_size = config['image_size']
        tasks = [(fig_size, R, Hp, timestep * 25) for R in config['R']
                 for Hp in config['Hp']]

        # Using multiprocessing Pool to parallelize the simulations
        # with Pool(processes=multiprocessing.cpu_count()) as pool:
        with Pool(processes=4) as pool:
            results = pool.map(worker_simulation, tasks)