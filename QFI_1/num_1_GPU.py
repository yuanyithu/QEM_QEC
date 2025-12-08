import numpy as np
from matplotlib import pyplot as plt
import sys
import multiprocessing as mp
from functools import partial

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: CuPy not installed, falling back to CPU")
    GPU_AVAILABLE = False


def generate_random_orthonormal_vectors_gpu(n: int, k: int, xp, device_id=None):
    """GPU版本的正交向量生成"""
    if device_id is not None and GPU_AVAILABLE:
        with cp.cuda.Device(device_id):
            N = 2**n
            M = 2**k
            A = xp.random.randn(N, M) + 1j * xp.random.randn(N, M)
            Q, _ = xp.linalg.qr(A)
            return Q
    else:
        N = 2**n
        M = 2**k
        A = xp.random.randn(N, M) + 1j * xp.random.randn(N, M)
        Q, _ = xp.linalg.qr(A)
        return Q


def noise_super_operator_gpu(p, xp):
    """GPU版本的噪声算符"""
    I2 = xp.array([[1+0j,0],[0,1]])
    X = xp.array([[0+0j,1],[1,0]])
    Y = xp.array([[0,-1j],[1j,0]])
    Z = xp.array([[1+0j,0],[0,-1]])
    return (1-p*3/4)*xp.einsum("ij,kl->ijkl",I2,I2) + p/4*(
        xp.einsum("ij,kl->ijkl",X,X)+
        xp.einsum("ij,kl->ijkl",Y,Y.conj())+
        xp.einsum("ij,kl->ijkl",Z,Z)
    )


def tensor_power_gpu(D, n, xp):
    """GPU版本的张量幂运算"""
    if n == 1:
        return xp.einsum("ijkl->ijkl", D)
    elif n % 2 == 0:
        D1 = tensor_power_gpu(D, n//2, xp)
        return xp.einsum("ijkl,abcd->iajbkcld", D1, D1).reshape([2**n,2**n,2**n,2**n])
    else:
        D1 = tensor_power_gpu(D, n//2, xp)
        return xp.einsum("ijkl,abcd,efgh->iaejbfkcgldh", D1, D1, D, optimize=True).reshape([2**n,2**n,2**n,2**n])


def generate_logical_hermitian_basis_gpu(Q, n, k, xp):
    """GPU版本的厄米基生成"""
    sigmas = []
    for a in range(2**k):
        for b in range(a+1, 2**k):
            outer_ab = xp.outer(Q[:,a], Q[:,b].conj())
            outer_ba = xp.outer(Q[:,b], Q[:,a].conj())
            sigma_x = (outer_ab + outer_ba) / xp.sqrt(2)
            sigma_y = (-1j*outer_ab + 1j*outer_ba) / xp.sqrt(2)
            sigmas.append(sigma_x)
            sigmas.append(sigma_y)

    outer_ii = xp.einsum("ai,bi->iab", Q, Q.conj())
    for m in range(2**k-1):
        sigma_z = xp.zeros([2**n, 2**n], dtype=xp.complex128)
        for i in range(m+1):
            sigma_z += outer_ii[i] / xp.sqrt((m+1)*(m+2))
        sigma_z -= (m+1) * outer_ii[m+1] / xp.sqrt((m+1)*(m+2))
        sigmas.append(sigma_z)
    return sigmas


def sample_gpu(n, k, p, N_til, xp, device_id=None):
    """GPU版本的单次采样"""
    Q = generate_random_orthonormal_vectors_gpu(n, k, xp, device_id)
    Pi = xp.einsum('ip, jp -> ij', Q, Q.conj())
    Pi_til = xp.einsum("ij,kl->ikjl", Pi, Pi.conj()).reshape([4**n, 4**n])
    N_Pi = xp.einsum("ij,j->i", N_til, Pi.reshape(4**n)).reshape([2**n, 2**n])
    R_N_Pi_til = 0.5 * (xp.kron(N_Pi, xp.eye(2**n)) + xp.kron(xp.eye(2**n), N_Pi.conj()))
    R_inverse_til = xp.linalg.pinv(R_N_Pi_til, rcond=1e-12)
    M = xp.einsum("ij,jk,kl,lm,mn->in", Pi_til, N_til, R_inverse_til, N_til, Pi_til, optimize=True)
    
    sigmas = generate_logical_hermitian_basis_gpu(Q, n, k, xp)
    J = xp.empty([4**k-1, 4**k-1], dtype=xp.complex128)
    for i in range(4**k-1):
        for j in range(4**k-1):
            J[i,j] = xp.einsum("i,ij,j->", sigmas[i].T.reshape(4**n), M, sigmas[j].reshape(4**n))
    
    eigs = xp.linalg.eigvalsh(J)
    return float(eigs.max())


def generate_gpu_worker(n, k, p, N, device_id):
    """在指定GPU上运行的工作函数"""
    if GPU_AVAILABLE:
        with cp.cuda.Device(device_id):
            xp = cp
            D = noise_super_operator_gpu(p, xp)
            N_til = xp.einsum("ijkl->ikjl", tensor_power_gpu(D, n, xp)).reshape([4**n, 4**n])
            
            result = []
            for i in range(N):
                result.append(sample_gpu(n, k, p, N_til, xp, device_id))
            
            # 确保GPU计算完成
            cp.cuda.Stream.null.synchronize()
            return n, p, result
    else:
        # CPU fallback
        xp = np
        D = noise_super_operator_gpu(p, xp)
        N_til = xp.einsum("ijkl->ikjl", tensor_power_gpu(D, n, xp)).reshape([4**n, 4**n])
        
        result = []
        for i in range(N):
            result.append(sample_gpu(n, k, p, N_til, xp, None))
        return n, p, result


def get_blues(n, start=0.35, stop=0.95):
    """从浅蓝到深蓝取 n 个颜色"""
    return plt.cm.Blues(np.linspace(start, stop, n))


def plot_one_p(ax, n_list, samples_by_n, medians, color, label, jitter=0.0):
    """绘制某个 p 的散点与中位数连线"""
    for n in n_list:
        y = samples_by_n[n]
        x = np.full_like(y, fill_value=n, dtype=float)
        if jitter:
            x = x + jitter * np.random.randn(len(x))
        ax.scatter(x, y, s=15, alpha=0.7, color=color, edgecolors='none')
    ax.plot(n_list, medians, marker='o', linewidth=2, color=color, label=label)


def worker_wrapper(args):
    """包装函数，用于多进程调用"""
    task, device_id = args
    n, k, p, N = task
    return generate_gpu_worker(n, k, p, N, device_id)


def plot_sweep_parallel(n_list, p_list, k, N, num_gpus=1, ylim=(0, 2), jitter=0.0):
    """多GPU并行计算并绘图"""
    print(f"Using {num_gpus} GPU(s)" if GPU_AVAILABLE else "Using CPU")
    
    # 创建所有任务
    tasks = [(n, k, p, N) for n in n_list for p in p_list]
    
    # 分配GPU设备ID（循环分配）
    device_ids = [i % num_gpus for i in range(len(tasks))]
    
    # 并行执行
    with mp.Pool(processes=min(len(tasks), num_gpus)) as pool:
        results = pool.map(worker_wrapper, [(task, device_id) for task, device_id in zip(tasks, device_ids)])
    
    # 整理结果
    data = {}
    for n, p, samples in results:
        if p not in data:
            data[p] = {}
        data[p][n] = np.array(samples)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = get_blues(len(p_list))
    
    for color, p in zip(colors, p_list):
        samples_by_n = data[p]
        maxs = [np.max(samples_by_n[n]) for n in n_list]
        plot_one_p(ax, n_list, samples_by_n, maxs, color, label=f"p={p:.1f}", jitter=jitter)
    
    ax.set_xlabel("n")
    ax.set_ylabel("eig(M)_max")
    ax.set_xticks(n_list)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="p", frameon=False)
    fig.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    k = 2
    N = 50
    n_list = [2, 3, 4,5,6]
    p_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    # 使用10张GPU并行计算
    plot_sweep_parallel(n_list, p_list, k, N, num_gpus=1, ylim=(0, 2), jitter=0.0)