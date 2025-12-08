import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from matplotlib import pyplot as plt
import sys
import multiprocessing as mp


def generate_random_orthonormal_vectors(n: int, k: int) -> np.ndarray:
    """
    高效生成 2^k 个随机正交归一的 2^n 维复向量。

    使用 QR 分解来确保向量集 (矩阵的列) 满足标准厄米特内积
    <u, v> = u.conj().T @ v 的正交归一性。

    返回:
    np.ndarray: 一个 (2^n, 2^k) 的复数数组 (complex128)，
                其列向量是互相正交归一的。
    """
    N = 2**n
    M = 2**k
    A = np.random.randn(N, M) + 1j * np.random.randn(N, M)
    Q, _ = np.linalg.qr(A)
    return Q

def noise_super_operator(p):
    """
    return一个2*2*2*2张量，实际上是向量化的单个qubit上的channel的某种表示
    """
    I2 = np.array([[1+0j,0],[0,1]])
    X = np.array([[0+0j,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1+0j,0],[0,-1]])
    return (1-p*3/4)*np.einsum("ij,kl->ijkl",I2,I2) + p/4*(np.einsum("ij,kl->ijkl",X,X)+np.einsum("ij,kl->ijkl",Y,Y.conj())+np.einsum("ij,kl->ijkl",Z,Z))

def tensor_power(D,n):
    """
    输入D是一个2*2*2*2张量
    返回n个qubit整体的向量化的channel，也就是[2**n,2**n,2**n,2**n]维张量
    """
    if n==1:
        return np.einsum("ijkl->ijkl",D)
    elif n%2 == 0:
        D1 = tensor_power(D,n//2)
        return np.einsum("ijkl,abcd->iajbkcld",D1,D1).reshape([2**n,2**n,2**n,2**n])
    elif n%2 == 1:
        D1 = tensor_power(D,n//2)
        return np.einsum("ijkl,abcd,efgh->iaejbfkcgldh",D1,D1,D,optimize=True).reshape([2**n,2**n,2**n,2**n])

def show_tensor(A):
    im = plt.imshow(A.real)
    plt.colorbar(im)
    plt.show()
    im = plt.imshow(A.imag)
    plt.colorbar(im)
    plt.show()

def generate_logical_hermitian_basis(Q,n,k):
    sigmas = []
    for a in range(2**k):
        for b in range(a+1,2**k):
            outer_ab = np.outer(Q[:,a],Q[:,b].conj())
            outer_ba = np.outer(Q[:,b],Q[:,a].conj())
            sigma_x = (outer_ab+outer_ba)/np.sqrt(2)
            sigma_y = (-1j*outer_ab+1j*outer_ba)/np.sqrt(2)
            sigmas.append(sigma_x)
            sigmas.append(sigma_y)

    outer_ii = np.einsum("ai,bi->iab",Q,Q.conj())
    for m in range(2**k-1):
        sigma_z = np.zeros([2**n,2**n])*(0+0j)
        for i in range(m+1):
            sigma_z += outer_ii[i]/np.sqrt((m+1)*(m+2))
        sigma_z -= (m+1)*outer_ii[m+1]/np.sqrt((m+1)*(m+2))
        sigmas.append(sigma_z)
    return sigmas

def sample(args):
    n, k, p, N_til = args
    Q = generate_random_orthonormal_vectors(n, k)
    Pi = np.einsum('ip, jp -> ij', Q, Q.conj())
    Pi_til = np.einsum("ij,kl->ikjl",Pi,Pi.conj()).reshape([4**n,4**n])
    N_Pi = np.einsum("ij,j->i",N_til,Pi.reshape(4**n)).reshape([2**n,2**n])
    R_N_Pi_til = 0.5*(np.kron(N_Pi,np.eye(2**n))+np.kron(np.eye(2**n),N_Pi.conj()))
    R_inverse_til = np.linalg.pinv(R_N_Pi_til,rcond=1e-12, hermitian=True)
    M = np.einsum("ij,jk,kl,lm,mn->in",Pi_til,N_til,R_inverse_til,N_til,Pi_til,optimize=True)
    sigmas = generate_logical_hermitian_basis(Q,n,k)
    J = np.empty([4**k-1,4**k-1])*(1+0j)
    for i in range(4**k-1):
        for j in range(4**k-1):
            J[i,j] = np.einsum("i,ij,j->",sigmas[i].T.reshape(4**n),M,sigmas[j].reshape(4**n))
    eigs = np.linalg.eigvalsh(J)
    return eigs.max()

def generate(n,k,p,N):
    D = noise_super_operator(p)
    N_til = np.einsum("ijkl->ikjl",tensor_power(D,n)).reshape([4**n,4**n])
    
    # 准备参数列表
    args_list = [(n, k, p, N_til) for _ in range(N)]
    
    # 使用多进程池
    num_processes = 16
    with mp.Pool(processes=num_processes) as pool:
        result = []
        for i, res in enumerate(pool.imap_unordered(sample, args_list)):
            result.append(res)
    
    return result

def get_blues(n, start=0.35, stop=0.95):
    """从浅蓝到深蓝取 n 个颜色。"""
    return plt.cm.Blues(np.linspace(start, stop, n))

def sample_for_p(n_list, k, p, N, gen_fn):
    """对固定 p，收集各 n 的 N 次采样结果。返回 {n: np.array}。"""
    return {n: np.asarray(gen_fn(n, k, p, N)) for n in n_list}

def maxs_from_samples(samples_by_n, n_list):
    """按 n_list 顺序给出最大值列表。"""
    return [np.max(samples_by_n[n]) for n in n_list]

def plot_one_p(ax, n_list, samples_by_n, medians, color, label, jitter=0.0):
    """绘制某个 p 的散点与中位数连线。"""
    for n in n_list:
        y = samples_by_n[n]
        x = np.full_like(y, fill_value=n, dtype=float)
        if jitter:
            x = x + jitter * np.random.randn(len(x))
        ax.scatter(x, y, s=15, alpha=0.7, color=color, edgecolors='none')
    ax.plot(n_list, medians, marker='o', linewidth=2, color=color, label=label)

def plot_sweep(n_list, p_list, k, N, gen_fn, ylim=(0, 2), jitter=0.0):
    """主绘图流程：遍历 p、绘制散点与最大值线。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = get_blues(len(p_list))
    for color, p in zip(colors, p_list):
        samples = sample_for_p(n_list, k, p, N, gen_fn)
        maxs = maxs_from_samples(samples, n_list)
        plot_one_p(ax, n_list, samples, maxs, color, label=f"p={p:.1f}", jitter=jitter)
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
    n_list = [2,3,4,5,6]
    p_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    plot_sweep(n_list, p_list, k, N, gen_fn=generate, ylim=(0, 2), jitter=0.0)