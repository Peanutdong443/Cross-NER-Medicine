import torch

def batch_trace(input_matrix, n, bs):
    a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
    b = a * input_matrix
    return torch.sum(torch.sum(b,-1),-1).unsqueeze(1)

def IPOT_distance_torch_batch_uniform(C, bs, n, m, iteration=50):
    C = C.float().cuda()
    T = IPOT_torch_batch_uniform(C, bs, n, m, iteration=iteration)
    temp = torch.bmm(torch.transpose(C,1,2), T)
    distance = batch_trace(temp, m, bs)
    return distance, T

# sinkhorn Algorithm
def IPOT_torch_batch_uniform(C, bs, n, m, beta=0.5, iteration=50):
    
    # C is the distance matrix
    # c: bs by n by m
    sigma = torch.ones(bs, int(m), 1).cuda()/float(m)
    T = torch.ones(bs, n, m).cuda()
    A = torch.exp(-C/beta).float().cuda()
    for t in range(iteration):
        Q = A * T # bs * n * m
        for k in range(1):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q,1,2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2,1)

    return T#.detach()

def GW_torch_batch(Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
    one_m = torch.ones(bs, m, 1).float().cuda()
    one_n = torch.ones(bs, n, 1).float().cuda()

    Cst = torch.bmm(torch.bmm(Cs**2, p), torch.transpose(one_m, 1, 2)) + \
          torch.bmm(one_n, torch.bmm(torch.transpose(q,1,2), torch.transpose(Ct**2, 1, 2))) # bs by n by m
    gamma = torch.bmm(p, q.transpose(2,1)) # outer product, init
    # gamma = torch.einsum('bi,bj->bij', (torch.squeeze(p), torch.squeeze(q))) # outer product, initialization
    for i in range(iteration):
        C_gamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
        # # Sinkhorn iteration
        # b = torch.ones(bs, m, 1).cuda()
        # K = torch.exp(-C_gamma/beta)
        # for i in range(50):cd
        #     a = p/(torch.bmm(K, b))
        #     b = q/torch.bmm(K.transpose(1,2), a)
        # gamma = a * K * b
        gamma = IPOT_torch_batch_uniform(C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
    Cgamma = Cst - 2 * torch.bmm(torch.bmm(Cs, gamma), torch.transpose(Ct, 1, 2))
    return gamma.detach(), Cgamma

def GW_distance_torch_batch_uniform(GS, GT, lamda=1e-1, iteration=5, OT_iteration=20):
    m = GS.size(1)
    n = GT.size(1)
    bs = 1
    p = (torch.ones(bs, m, 1)/m).cuda()
    q = (torch.ones(bs, n, 1)/n).cuda()

    GS = GS.float().cuda()
    GT = GT.float().cuda()

    m = GT.size(2)
    n = GS.size(2)
    T, Cst = GW_torch_batch(GS, GT, bs, n, m, p, q, beta=lamda, iteration=iteration, OT_iteration=OT_iteration)
    temp = torch.bmm(torch.transpose(Cst,1,2), T)
    distance = batch_trace(temp, m, bs)

    return distance, T

def cost_matrix_batch_torch(x, y, beta=0.1):
    """
    input:
        X: batch_size*C1*D tensor
        Y: batch_size*C2*D tensor
        beta: threshold for small distance
    return:
        cosine_matrix: batch_size*C1*C2 tensor
        cos_sim: batch_size*C1*C2 tensor
    """
    batch_size = x.size(0)
    assert(x.size(0)==y.size(0))
    assert(x.size(2)==y.size(2))

    x = x.div(torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
    y = y.div(torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)
    cos_sim = torch.bmm(x, torch.transpose(y, 1, 2))
    cos_dist = 1 - cos_sim

    min_score = cos_dist.min()
    max_score = cos_dist.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_dist - threshold)

    return cos_dist, cos_sim

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist