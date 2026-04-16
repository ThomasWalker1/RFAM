import torch
import torch.nn.functional as F
import utils.classic_kernel as classic_kernel
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def laplace_kernel_M(pair1, pair2, bandwidth, M):
    return classic_kernel.laplacian_M(pair1, pair2, bandwidth, M)


def get_grads(X, sol, L, P, compute_metrics=False):
    start = time.time()
    num_samples = 20000
    
    # Generate random indices natively on the device
    indices = torch.randint(0, len(X), (min(len(X), num_samples),), device=device)
    x = X[indices, :]

    K = laplace_kernel_M(X, x, L, P)

    dist = classic_kernel.euclidean_distances_M(X, x, P, squared=False)
    dist = torch.where(dist < 1e-10, torch.zeros(1, device=device).float(), dist)

    K = K / dist
    K[torch.isinf(K)] = 0.

    a1 = sol.T.clone()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = (a1 @ X1).reshape(-1, c*d)
    del a1, X1

    step2 = (K.T @ step1).reshape(-1, c, d)
    del step1

    a2 = sol.clone()
    step3 = (a2 @ K).T
    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * (-1.0 / L)
    
    # Keep accumulator on GPU to avoid CPU synchronization in the loop
    M_accum = torch.zeros((d, d), device=device)
    
    if compute_metrics:
        total_effective_rank = torch.zeros(1, device=device)
        total_cos_sim = torch.zeros(1, device=device)

    bs = 10
    batches = torch.split(G, bs)
    
    if compute_metrics:
        x_batches = torch.split(x, bs)

    for i in range(len(batches)):
        grad = batches[i] # Already on GPU
        gradT = torch.transpose(grad, 1, 2)
        M_accum += torch.sum(gradT @ grad, dim=0)
        
        if compute_metrics:
            x_batch = x_batches[i]
            U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
            
            stable_rank = (torch.sum(S, dim=-1)**2) / torch.sum(S**2, dim=-1)
            total_effective_rank += torch.sum(stable_rank)
            
            v1 = Vh[:, 0, :] 
            cos_sim = torch.abs(F.cosine_similarity(v1, x_batch, dim=-1))
            total_cos_sim += torch.sum(cos_sim)

        del grad, gradT
        
    # Average M and optionally transfer to CPU at the very end if required by upstream code
    M_accum /= len(G)

    end = time.time()

    if compute_metrics:
        avg_effective_rank = (total_effective_rank / len(G)).item()
        avg_cos_sim = (total_cos_sim / len(G)).item()
        return M_accum, avg_effective_rank, avg_cos_sim

    return M_accum

def convert_one_hot(y, c):
    # PyTorch native one-hot encoding
    return F.one_hot(y.to(torch.int64), num_classes=c).float()

def hyperparam_train(X_train, y_train, X_test, y_test, c,
                     iters=5, reg=0, L=10, normalize=False,
                     alpha=0.1):

    # Move initial data to device
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, device=device)

    y_train = convert_one_hot(y_train, c)
    y_test = convert_one_hot(y_test, c)

    if normalize:
        X_train /= torch.norm(X_train, dim=-1, keepdim=True)
        X_test /= torch.norm(X_test, dim=-1, keepdim=True)

    best_acc = 0.
    best_iter = 0
    best_M = None

    n, d = X_train.shape
    
    # Initialize M as ridge-blended covariance (RFAM initialization)
    cov_matrix = (X_train.T @ X_train) / (n - 1)
    identity = torch.eye(d, dtype=torch.float32, device=device)
    M = (1.0 - alpha) * cov_matrix + alpha * identity
    trace = torch.trace(M)
    if trace > 1e-10:
        M = M * (d / trace)

    for i in range(iters):
        K_train = laplace_kernel_M(X_train, X_train, L, M)
        reg_matrix = reg * torch.eye(K_train.size(0), device=device)
        
        # GPU accelerated linear solve
        sol = torch.linalg.solve(K_train + reg_matrix, y_train).T

        K_test = laplace_kernel_M(X_train, X_test, L, M)
        preds = (sol @ K_test).T

        preds_argmax = torch.argmax(preds, dim=-1)
        labels = torch.argmax(y_test, dim=-1)
        count = torch.sum(labels == preds_argmax).item()

        old_test_acc = count / len(labels)

        if old_test_acc > best_acc:
            best_iter = i
            best_acc = old_test_acc
            best_M = M.clone()
            
        M = get_grads(X_train, sol, L, M, compute_metrics=False)
        trace = torch.trace(M)
        if trace > 1e-10:
            M = M * (d / trace)

    return best_acc, best_iter, best_M.cpu().numpy() if best_M is not None else None

def pgd_attack(X_train, X_test, y_test_labels, sol, L, M, epsilon=0.05, alpha=0.01, iters=20, normalize=False):
    # Ensure variables are cloned and on device
    sol_t = sol.clone().to(device)
    M_t = M.clone().to(device)
    
    X_adv = X_test.clone().detach() + torch.empty_like(X_test).uniform_(-epsilon, epsilon)
    
    for i in range(iters):
        X_adv.requires_grad_(True)
        
        if normalize:
            row_norms = torch.norm(X_adv, p=2, dim=-1, keepdim=True)
            row_norms = torch.clamp(row_norms, min=1e-10)
            X_adv_norm = X_adv / row_norms
        else:
            X_adv_norm = X_adv
            
        K_adv = laplace_kernel_M(X_train, X_adv_norm, L, M_t)
        preds = (sol_t @ K_adv).T
        y_test_one_hot = F.one_hot(y_test_labels, num_classes=preds.shape[-1]).float()
        loss = F.mse_loss(preds, y_test_one_hot)
        
        loss.backward()
        
        with torch.no_grad():
            X_adv_next = X_adv + alpha * X_adv.grad.sign()
            eta = torch.clamp(X_adv_next - X_test, min=-epsilon, max=epsilon)
            X_adv = (X_test + eta).detach() 
        
    return X_adv

def train(X_train, y_train, X_test, y_test, c, M,
          iters=5, reg=0, L=10, normalize=False, epsilons=[0.05,0.1,0.2,0.4,1.0]): 

    # Move to Device immediately
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, device=device)
    M = torch.tensor(M, dtype=torch.float32, device=device)

    y_train = convert_one_hot(y_train, c)
    y_test = convert_one_hot(y_test, c)

    if normalize:
        X_train /= torch.norm(X_train, dim=-1, keepdim=True)
        X_test /= torch.norm(X_test, dim=-1, keepdim=True)

    # --- Standard Training ---
    K_train = laplace_kernel_M(X_train, X_train, L, M)
    reg_matrix = reg * torch.eye(K_train.size(0), device=device)
    sol = torch.linalg.solve(K_train + reg_matrix, y_train).T

    K_test = laplace_kernel_M(X_train, X_test, L, M)
    preds = (sol @ K_test).T

    preds_argmax = torch.argmax(preds, dim=-1)
    labels = torch.argmax(y_test, dim=-1)
    count = torch.sum(labels == preds_argmax).item()

    acc = count / len(labels)
    
    # --- Adversarial ---
    X_test_advs = {epsilon: pgd_attack(
        X_train=X_train, 
        X_test=X_test, 
        y_test_labels=labels, 
        sol=sol, 
        L=L, 
        M=M,
        alpha=epsilon/10,
        epsilon=epsilon,
        normalize=normalize
    ) for epsilon in epsilons}
    
    adv_robust_accs={}
    for epsilon, X_test_adv in X_test_advs.items():
        K_test_adv = laplace_kernel_M(X_train, X_test_adv, L, M)
        preds_adv = (sol @ K_test_adv).T
        
        preds_adv_argmax = torch.argmax(preds_adv, dim=-1)
        count_adv = torch.sum(labels == preds_adv_argmax).item()
        
        adv_robust_accs[epsilon] = count_adv / len(labels)

    _, eff_rank, cos_sim = get_grads(X_train, sol, L, M, compute_metrics=True)
    
    return acc, adv_robust_accs, eff_rank, cos_sim