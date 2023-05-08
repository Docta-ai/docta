# HOC estimator

import numpy as np
import torch
import time
from tqdm import tqdm


from .core_utils import cosDistance

smp = torch.nn.Softmax(dim=0)
smt = torch.nn.Softmax(dim=1)


def consensus_analytical(cfg, T, P, mode):
    r""" Compute the first-, second-, and third-order of concensus matrices.
    Args:
        cfg: configuration
        T : noise transition matrix
        P : the priors of P(Y = i), i \in [K]
        mode :
    Return:
        c_analytical[0] : first-order concensus
        c_analytical[1] : second-order concensus
        c_analytical[2] : third-order concensus 
    """

    KINDS = cfg.num_classes
    P = P.reshape((KINDS, 1))
    c_analytical = [[] for _ in range(3)]

    c_analytical[0] = torch.mm(T.transpose(0, 1), P).transpose(0, 1)
    c_analytical[2] = torch.zeros((KINDS, KINDS, KINDS))

    temp33 = torch.tensor([])
    for i in range(KINDS):
        Ti = torch.cat((T[:, i:], T[:, :i]), 1)
        temp2 = torch.mm((T * Ti).transpose(0, 1), P)
        c_analytical[1] = torch.cat(
            [c_analytical[1], temp2], 1) if i != 0 else temp2

        for j in range(KINDS):
            Tj = torch.cat((T[:, j:], T[:, :j]), 1)
            temp3 = torch.mm((T * Ti * Tj).transpose(0, 1), P)
            temp33 = torch.cat([temp33, temp3], 1) if j != 0 else temp3
        # adjust the order of the output (N*N*N), keeping consistent with c_est
        t3 = []
        for p3 in range(KINDS):
            t3 = torch.cat((temp33[p3, KINDS - p3:], temp33[p3, :KINDS - p3]))
            temp33[p3] = t3
        if mode == -1:
            for r in range(KINDS):
                c_analytical[2][r][(i+r+KINDS) % KINDS] = temp33[r]
        else:
            c_analytical[2][mode][(i + mode + KINDS) % KINDS] = temp33[mode]

    # adjust the order of the output (N*N), keeping consistent with c_est
    temp = []
    for p1 in range(KINDS):
        temp = torch.cat(
            (c_analytical[1][p1, KINDS-p1:], c_analytical[1][p1, :KINDS-p1]))
        c_analytical[1][p1] = temp
    return c_analytical


def func(cfg, c_est, T_out, P_out):
    """ Compute the loss of estimated concensus matrices
    """
    hoc_cfg = cfg.hoc_cfg
    loss = torch.tensor(0.0).to(hoc_cfg.device)  # initialize the loss

    P = smp(P_out)
    T = smt(T_out)

    # mode = random.randint(0, cfg.num_classes - 1) # random update for speedup
    mode = -1  # calculate all patterns

    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at
    # this time: N, N*N, N*N*N
    c_ana = consensus_analytical(
        cfg, T.to(hoc_cfg.device), P.to(hoc_cfg.device), mode)

    # weight for differet orders of concensus patterns
    weight = [1.0, 1.0, 1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        c_ana[j] = c_ana[j].to(hoc_cfg.device)
        loss += weight[j] * torch.norm(c_est[j] - c_ana[j])  # / np.sqrt(N**j)

    return loss


def calc_func(cfg, c_est):
    """ Optimize over the noise transition matrix T and prior P
    """

    N = cfg.num_classes
    hoc_cfg = cfg.hoc_cfg
    hoc_cfg.device = torch.device(hoc_cfg.device)
    if hoc_cfg.T0 is None:
        T = 5 * torch.eye(N) - torch.ones((N, N))
    else:
        T = hoc_cfg.T0

    if hoc_cfg.p0 is None:
        P = torch.ones((N, 1)) / N + torch.rand((N, 1)) * 0.1
    else:
        P = hoc_cfg.p0

    T = T.to(hoc_cfg.device)
    P = P.to(hoc_cfg.device)
    c_est = [item.to(hoc_cfg.device) for item in c_est]
    print(f'Use {hoc_cfg.device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr=hoc_cfg.lr)

    # train
    loss_min = 100.0
    T_rec = T.detach()
    P_rec = P.detach()

    time1 = time.time()
    # use gradient descent to solve consensus equations
    for step in tqdm(range(hoc_cfg.max_step)):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(cfg, c_est, T, P)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()

        if cfg.details:  # print log
            if step % 100 == 0:
                print('loss {}'.format(loss))
                print(f'step: {step}  time_cost: {time.time() - time1}')
                print(
                    f'T {np.round(smt(T.cpu()).detach().numpy()*100,1)}', flush=True)
                print(
                    f'P {np.round(smp(P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
                time1 = time.time()
    print(f'Solve equations... [Done]')
    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def get_consensus_patterns(dataset, sample, k=1+2):
    """ KNN estimation
    """
    feature = dataset.feature if isinstance(
        dataset.feature, torch.Tensor) else torch.tensor(dataset.feature)
    label = dataset.label if isinstance(
        dataset.label, torch.Tensor) else torch.tensor(dataset.label)
    feature = feature[sample]
    label = label[sample]
    dist = cosDistance(feature.float())
    values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
    knn_labels = label[indices]
    return knn_labels, values


def consensus_counts(cfg, consensus_patterns):
    """ Count the consensus
    """
    KINDS = cfg.num_classes

    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)

    for _, pattern in enumerate(consensus_patterns):
        cnt[0][pattern[0]] += 1
        cnt[1][pattern[0]][pattern[1]] += 1
        cnt[2][pattern[0]][pattern[1]][pattern[2]] += 1

    return cnt


def estimator_hoc(cfg, dataset):
    """ HOC estimator
    """
    print('Estimating consensus patterns...')

    KINDS = cfg.num_classes

    # initialize sample counts
    c_est = [[] for _ in range(3)]
    c_est[0] = torch.zeros(KINDS)
    c_est[1] = torch.zeros(KINDS, KINDS)
    c_est[2] = torch.zeros(KINDS, KINDS, KINDS)

    sample_size = int(len(dataset) * 0.9)

    if cfg.hoc_cfg is not None and cfg.hoc_cfg.sample_size:
        sample_size = np.min((cfg.hoc_cfg.sample_size, int(len(dataset)*0.9)))

    for idx in tqdm(range(cfg.hoc_cfg.num_rounds)):
        if cfg.details:
            print(idx, flush=True)

        sample = np.random.choice(
            range(len(dataset)), sample_size, replace=False)

        if not cfg.hoc_cfg.already_2nn:
            consensus_patterns_sample, _ = get_consensus_patterns(
                dataset, sample)
        else:
            consensus_patterns_sample = torch.tensor(dataset.consensus_patterns[sample]) if isinstance(
                dataset.consensus_patterns, list) else dataset.consensus_patterns[sample]
        cnt_y_3 = consensus_counts(cfg, consensus_patterns_sample)
        for i in range(3):
            cnt_y_3[i] /= cfg.hoc_cfg.sample_size
            c_est[i] = c_est[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    print('Estimating consensus patterns... [Done]')

    for j in range(3):
        c_est[j] = c_est[j] / cfg.hoc_cfg.num_rounds

    loss_min, T_est, p_est, T_est_before_sfmx = calc_func(cfg, c_est)

    T_est = T_est.cpu().numpy()
    T_est_before_sfmx = T_est_before_sfmx.cpu().numpy()
    p_est = p_est.cpu().numpy()
    return T_est, p_est, T_est_before_sfmx
