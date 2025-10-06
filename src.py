import torch
import torch.nn as nn
from tqdm import tqdm
import gurobipy as gp
from gurobipy import GRB
import pulp as pl
import pyscipopt as scip
import numpy as np
from scipy import sparse as sp


class MFC():
    """
    Minimum finite covering (MFC)
    """
    def __init__(self, data, norm=2, num_lim=1e10):
        """
        Inputs:
            data: (n, *) tensor, n: number of data points, *: dimension of data points
            norm: norm of distance, 1, 2, inf
            num_lim: number of data points to be considered
        """
        self.data = data
        self.norm = norm
        self.num_lim = num_lim

    def distanceMatrix(self):
        """
        Generate distance matrix for given dataset
        """
        n = self.data.shape[0]
        n = min(n, self.num_lim)
        A = torch.zeros((n, n))
        print('Generate distance matrix for given dataset:')
        for i in tqdm(range(n)):
            for j in range(i, n):
                A[i, j] = torch.norm(self.data[i] - self.data[j], p=self.norm)
                A[j, i] = A[i, j]

        return A

    def mfc_gurobi_continuous(self, A, eta=0):
        """
        LP relaxation of minimum finite covering problem
        Inputs:
            A: distance matrix
            eta: radius of covering balls
        """
        A_adj = sp.csr_matrix(A <= eta)
        n1, n2 = A_adj.shape
        m = gp.Model("LP")
        m.setParam("OutputFlag", 0)
        s = m.addMVar(shape=n2, vtype=GRB.CONTINUOUS, name="s")
        obj = s.sum()
        m.setObjective(obj, GRB.MINIMIZE)
        m.addConstr(A_adj @ s >= np.ones(n1), name="c")

        m.optimize()

        return s.X, m.Runtime

    def mfc(self, A, eta=0, k=False, solver='gurobi_relax'):
        """
        Solve minimum finite covering problem
        Inputs:
            A: distance matrix
            eta: radius of covering balls
            k: number of covering balls
            solver: choice of solvers
                "gurobi": GUROBI solver (https://www.gurobi.com/)
                "gurobi_relax": GUROBI with LP relaxation
                "cbc": CBC solver (https://github.com/coin-or/Cbc) from PuLP (https://github.com/coin-or/pulp) interface
                "scip": SCIP solver (https://www.scipopt.org/)
        """
        if solver == 'gurobi':
            A_adj = sp.csr_matrix(A <= eta)
            n1, n2 = A_adj.shape
            m = gp.Model("MILP")
            m.setParam("OutputFlag", 0)
            m.setParam("TimeLimit", 3000)
            s = m.addMVar(shape=n2, vtype=GRB.BINARY, name="s")
            obj = s.sum()
            m.setObjective(obj, GRB.MAXIMIZE)
            m.addConstr(A_adj @ s >= np.ones(n1), name="c")
            if k:
                m.addConstr(obj == k, name='c0')
            m.optimize()

            if m.status == 2:  # 2 optimal; 3 infeasible; 4 infeasible or unbounded; 5 unbounded; 9 time_limit; 12 numeric; 13 suboptimal; 14 inprogress; 17 mem_limit.
                return s.X, m.Runtime
            else:
                return None, m.Runtime
        elif solver == 'gurobi_relax':
            n = A.shape[0]
            l = list(range(0, n, 5000))
            if len(l) > 1:
                l[-1] = n
            else:
                l = [0, n]
            sol_total = np.zeros(n)
            t_total = 0
            for i in range(len(l) - 1):
                sol, t = self.mfc_gurobi_continuous(A[l[i]:l[i + 1], :], eta)
                sol_total += sol
                t_total += t
            idx = sol_total.nonzero()[0]
            if len(idx) <= k:
                return np.ceil(sol_total), t_total  # m.ObjVal
            else:
                A_adj = sp.csr_matrix(A[:, idx] <= eta)
                n1, n2 = A_adj.shape
                m = gp.Model("MILP")
                m.setParam("OutputFlag", 0)
                m.setParam("TimeLimit", 3000)
                s = m.addMVar(shape=n2, vtype=GRB.BINARY, name="s")
                obj = s.sum()
                m.setObjective(obj, GRB.MINIMIZE)
                m.addConstr(A_adj @ s >= np.ones(n1), name="c")
                if k:
                    m.addConstr(obj == k, name='c0')
                m.optimize()

                if m.status == 2:  # 2 optimal; 3 infeasible; 4 infeasible or unbounded; 5 unbounded; 9 time_limit; 12 numeric; 13 suboptimal; 14 inprogress; 17 mem_limit.
                    v = np.zeros(n1)
                    for i, id in enumerate(idx):
                        v[id] = s.X[i]
                    return v, t + m.Runtime  # m.ObjVal
                else:
                    return None, t + m.Runtime
        elif solver == 'cbc':
            A_adj = sp.csr_matrix(A <= eta)
            n1, n2 = A_adj.shape
            m = pl.LpProblem("MILP", pl.LpMinimize)
            s = pl.LpVariable.dicts("s", range(n2), cat=pl.LpBinary)
            m += pl.lpSum([s[j] for i in range(n1) for j in A_adj[i].indices])
            for i in range(n1):
                indices = A_adj[i].indices
                m += pl.lpSum([s[j] for j in indices]) >= 1, f"c_{i}"
            if k:
                m += pl.lpSum([s[j] for j in range(n2)]) == k, "c0"
            solver = pl.PULP_CBC_CMD(msg=0, timeLimit=3000)
            status = m.solve(solver)
            if status == pl.LpStatusOptimal:
                sol = np.array([pl.value(s[j]) for j in range(n2)])
                return sol, m.solutionTime
            else:
                return None, m.solutionTime
        elif solver == 'scip':
            A_adj = sp.csr_matrix(A <= eta)
            n1, n2 = A_adj.shape
            m = scip.Model("MILP")
            m.setPresolve(0)  # optional: turn off aggressive presolve if needed
            m.setParam("limits/time", 3000)
            m.hideOutput()  # suppress solver output
            s = {}
            for j in range(n2):
                s[j] = m.addVar(vtype="B", name=f"s_{j}")
            m.setObjective(scip.quicksum(s[j] for j in range(n2)), "maximize")
            for i in range(n1):
                indices = A_adj[i].indices
                m.addCons(scip.quicksum(s[j] for j in indices) >= 1, name=f"c_{i}")
            if k:
                m.addCons(scip.quicksum(s[j] for j in range(n2)) == k, name="c0")
            m.optimize()
            status = m.getStatus()
            if status == "optimal":
                sol = np.array([m.getVal(s[j]) for j in range(n2)])
                runtime = m.getSolvingTime()
                return sol, runtime
            else:
                return None, m.getSolvingTime()

    def gen_data(self, A, eta=0, k=False, save=True):
        """
        Generate data points for minimum finite covering
        Inputs:
            A: distance matrix
            eta: radius of covering balls
            k: number of covering balls
            save: save generated data points
        """
        t_total = 0
        if k:
            eta_l, eta_u = 0, torch.max(A)
            print('Initial finite covering (enclosing ball) with k={:.0f}, eta={:.3f}.'.format(k, eta_u))
            i = 0
            # while i <= 10:
            while eta_u - eta_l > 1e-3:
                i += 1
                print('Set bounds for eta: [%.3f, %.3f], gap: %.3f.' % (eta_l, eta_u, eta_u - eta_l))
                eta = (eta_l + eta_u) / 2
                sol, t = self.mfc(A, eta, k, solver=solver)
                t_total += t
                if sol is None:
                    print('Iter {}: \nNo feasible finite covering for fixed k={:.0f} and fixed eta={:.3f}.'.format(i, k,
                                                                                                                   eta))
                    eta_l = eta
                else:
                    print('Iter {}: \nFound feasible finite covering for fixed k={:.0f} and fixed eta={:.3f}.'.format(i,
                                                                                                                      k,
                                                                                                                      eta))
                    eta_u = eta
            eta = eta_u
            sol, t = self.mfc(A, eta, k, solver=solver)
            t_total += t
            print('Minimum finite covering for fixed k={:.0f}, got eta={:.3f}.\n'.format(k, eta))
        else:
            sol, t = self.mfc(A, eta, solver=solver)
            t_total += t
            if sol is None:
                raise Exception('No feasible finite covering for fixed eta={:.3f}.'.format(eta))
            else:
                print('Minimum finite covering for fixed radius eta={:.3f}, got k={:.0f}.'.format(eta, sol[0].sum()))

        idx = torch.nonzero(torch.tensor(sol)).reshape(-1)
        sub_data = torch.zeros(0)
        for i in range(len(idx)):
            sub_data = torch.cat((sub_data, self.data[idx[i]:idx[i] + 1]), 0)

        if save:
            with open('data.npy', 'wb') as f:
                torch.save(f, sub_data)

        return sub_data, eta, sol, t_total


class TR():
    """
    Standard and Adversarial Training
    """
    def __init__(self, model, train_loader, test_loader, eps_linf, eps_l2):
        """
        Inputs:
            model: neural network model
            train_loader: training data loader
            test_loader: test data loader
            eps_linf: radius of l_inf ball
            eps_l2: radius of l_2 ball
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pgd_linf = eps_linf
        self.pgd_l2 = eps_l2

    def epoch(self, loader, weight=False, opt=None):
        """
        Standard training/evaluation epoch over the dataset
        Inputs:
            loader: data loader
            weight: whether to use weighted loss
            opt: 'None' for evaluation, otherwise for training
        """
        acc, l = 0., 0.
        for data in loader:
            if weight:
                X, y, w = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                yp = self.model(X)
                loss = (nn.CrossEntropyLoss(reduction='none')(yp, y) @ w).mean()
            else:
                X, y = data[0].to(self.device), data[1].to(self.device)
                yp = self.model(X)
                loss = nn.CrossEntropyLoss()(yp, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            acc += (yp.max(dim=1)[1] == y).sum().item()
            l += loss.item() * X.shape[0]
        return acc / len(loader.dataset), l / len(loader.dataset)


    def pgd_linf(self, X, y, epsilon=0.1, alpha=0.1, num_iter=20, randomize=False):
        """
        Construct PGD-linf adversarial examples on the examples X
        Inputs:
            X: input tensor
            y: label tensor
            epsilon: radius of l_inf ball
            alpha: learning rate of inner maximization
            num_iter: number of iterations of inner maximization
            randomize: whether to randomize the initial perturbation
        """
        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
        else:
            delta = torch.zeros_like(X, requires_grad=True)

        for _ in range(num_iter):
            loss = nn.CrossEntropyLoss()(self.model(X + delta), y)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()


    def norms(self, Z):
        """
        Compute the norms of the input tensor Z
        """
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


    def pgd_l2(self, X, y, epsilon=0.1, alpha=0.1, num_iter=40):
        """
        Construct PGD-l2 adversarial examples on the examples X
        Inputs:
            X: input tensor
            y: label tensor
            epsilon: radius of l_2 ball
            alpha: learning rate of inner maximization
            num_iter: number of iterations of inner maximization
        """
        delta = torch.zeros_like(X, requires_grad=True)
        for _ in range(num_iter):
            loss = nn.CrossEntropyLoss()(self.model(X + delta), y)
            loss.backward()
            delta.data += alpha * delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -X), 1 - X)  # clip X+delta to [0,1]
            delta.data *= epsilon / self.norms(delta.detach()).clamp(min=epsilon.to(self.device))
            delta.grad.zero_()

        return delta.detach()


    def epoch_adv(self, loader, attack, weight=False, opt=None, **kwargs):
        """
        Adversarial training/evaluation epoch over the dataset
        Inputs:
            loader: data loader
            attack: adversarial attack
            weight: whether to use weighted loss
            opt: 'None' for evaluation, otherwise for training
            kwargs: parameters for adversarial attack
        """
        acc, l = 0., 0.
        for data in loader:
            if weight:
                X, y, w = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                delta = attack(self.model, X, y, **kwargs)
                yp = self.model(X + delta)
                loss = (nn.CrossEntropyLoss(reduction='none')(yp, y) @ w).mean()
            else:
                X, y = data[0].to(self.device), data[1].to(self.device)
                delta = attack(self.model, X, y, **kwargs)
                yp = self.model(X + delta)
                loss = nn.CrossEntropyLoss()(yp, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            acc += (yp.max(dim=1)[1] == y).sum().item()
            l += loss.item() * X.shape[0]

        return acc / len(loader.dataset), l / len(loader.dataset)


    def train(self, weight=False, optimizer='SGD', itr=100, lr=1e-3):
        """
        Standard training
        Inputs:
            weight: whether to use weighted loss
            optimizer: optimizer, 'SGD' or 'Adam'
            itr: number of iterations
            lr: learning rate
        """
        if optimizer == 'SGD':
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif optimizer == 'Adam':
            opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        total_train_acc, total_train_loss = [], []
        total_test_acc, total_test_loss = [], []
        total_linf_adv_acc, total_linf_adv_loss = [], []
        total_l2_adv_acc, total_l2_adv_loss = [], []
        for _ in tqdm(range(itr)):
            train_acc, train_loss = self.epoch(self.train_loader, self.model, weight=weight, opt=opt)
            total_train_acc.append(train_acc)
            total_train_loss.append(train_loss)

            test_acc, test_loss = self.epoch(self.test_loader, self.model)
            total_test_acc.append(test_acc)
            total_test_loss.append(test_loss)

            linf_adv_acc, linf_adv_loss = self.epoch_adv(self.test_loader, self.model, self.pgd_linf, epsilon=self.eps_linf)
            total_linf_adv_acc.append(linf_adv_acc)
            total_linf_adv_loss.append(linf_adv_loss)

            l2_adv_acc, l2_adv_loss = self.epoch_adv(self.test_loader, self.model, self.pgd_l2, epsilon=self.eps_l2)
            total_l2_adv_acc.append(l2_adv_acc)
            total_l2_adv_loss.append(l2_adv_loss)
        return self.model, total_train_acc, total_train_loss, total_test_acc, total_test_loss, total_linf_adv_acc, total_linf_adv_loss, total_l2_adv_acc, total_l2_adv_loss


    def train_adv(self, attack, eps, weight=False, optimizer='SGD', itr=100, lr=1e-3, alpha=1e-1):
        """
        Adversarial training
        Inputs:
            attack: adversarial attack
            eps: radius of adversarial ball
            weight: whether to use weighted loss
            optimizer: optimizer, 'SGD' or 'Adam'
            itr: number of iterations
            lr: learning rate of outer minimization
            alpha: learning rate of inner maximization
        """
        if optimizer == 'SGD':
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        elif optimizer == 'Adam':
            opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        total_train_acc, total_train_loss = [], []
        total_test_acc, total_test_loss = [], []
        total_linf_adv_acc, total_linf_adv_loss = [], []
        total_l2_adv_acc, total_l2_adv_loss = [], []
        for _ in tqdm(range(itr)):
            train_acc, train_loss = self.epoch_adv(self.train_loader, self.model, attack, weight=weight, opt=opt, epsilon=eps, alpha=alpha)
            total_train_acc.append(train_acc)
            total_train_loss.append(train_loss)

            test_acc, test_loss = self.epoch(self.test_loader, self.model)
            total_test_acc.append(test_acc)
            total_test_loss.append(test_loss)

            linf_adv_acc, linf_adv_loss = self.epoch_adv(self.test_loader, self.model, self.pgd_linf, epsilon=self.eps_linf)
            total_linf_adv_acc.append(linf_adv_acc)
            total_linf_adv_loss.append(linf_adv_loss)

            l2_adv_acc, l2_adv_loss = self.epoch_adv(self.test_loader, self.model, self.pgd_l2, epsilon=self.eps_l2)
            total_l2_adv_acc.append(l2_adv_acc)
            total_l2_adv_loss.append(l2_adv_loss)
        return self.model, total_train_acc, total_train_loss, total_test_acc, total_test_loss, total_linf_adv_acc, total_linf_adv_loss, total_l2_adv_acc, total_l2_adv_loss
