import jax.numpy as jnp
from jax import grad, jit
from jax import random
from jax.random import PRNGKey

def generate_random_orthonormal_vectors_jax(key: PRNGKey, n: int, k: int) -> jnp.ndarray:
    """
    使用 JAX 生成一个随机的（复数）正交向量组。

    参数:
        key: JAX 随机数生成器 (PRNG) 密钥。
        n: 维度 N = 2**n。
        k: 向量数量 M = 2**k。

    返回:
        一个形状为 (N, M) 或 (N, N) 的 JAX 数组 Q（取决于 N 和 M 的比较），
        其列是正交的 (Q^H Q = I)。
    """
    N = 2**n
    M = 2**k    
    # JAX 需要显式地分割 key 来生成两个独立的随机数
    key_real, key_imag = random.split(key)

    # 1. 生成实部，服从 N(0, 1) 分布
    real_part = random.normal(key_real, shape=(N, M))
    # 2. 生成虚部，服从 N(0, 1) 分布
    imag_part = random.normal(key_imag, shape=(N, M))
    # 组合成复数矩阵 A
    A = real_part + 1j * imag_part
    # 3. 执行 QR 分解
    Q, _ = jnp.linalg.qr(A)
    return Q

def trivial_Q(n):
    Q_zeros = jnp.zeros([2**n,2],dtype=jnp.complex64)
    Q = (Q_zeros.at[0,0].set(1)).at[1,1].set(1)
    return Q

def noise_super_operator(p):
    """
    return一个2*2*2*2张量，实际上是向量化的单个qubit上的channel的某种表示
    """
    I2 = jnp.array([[1+0j,0],[0,1]])
    X = jnp.array([[0+0j,1],[1,0]])
    Y = jnp.array([[0,-1j],[1j,0]])
    Z = jnp.array([[1+0j,0],[0,-1]])
    return (1-p*3/4)*jnp.einsum("ij,kl->ijkl",I2,I2) + p/4*(jnp.einsum("ij,kl->ijkl",X,X)+jnp.einsum("ij,kl->ijkl",Y,Y.conj())+jnp.einsum("ij,kl->ijkl",Z,Z))

def tensor_power(D,n):
    """
    输入D是一个2*2*2*2张量
    返回n个qubit整体的向量化的channel，也就是[2**n,2**n,2**n,2**n]维张量
    """
    if n==1:
        return D
    elif n%2 == 0:
        D1 = tensor_power(D,n//2)
        return jnp.einsum("ijkl,abcd->iajbkcld",D1,D1).reshape([2**n,2**n,2**n,2**n])
    elif n%2 == 1:
        D1 = tensor_power(D,n//2)
        return jnp.einsum("ijkl,abcd,efgh->iaejbfkcgldh",D1,D1,D,optimize=True).reshape([2**n,2**n,2**n,2**n])

def generate_logical_hermitian_basis(Q,n,k):
    sigmas = []
    for a in range(2**k):
        for b in range(a+1,2**k):
            outer_ab = jnp.outer(Q[:,a],Q[:,b].conj())
            outer_ba = jnp.outer(Q[:,b],Q[:,a].conj())
            sigma_x = (outer_ab+outer_ba)/jnp.sqrt(2)
            sigma_y = (-1j*outer_ab+1j*outer_ba)/jnp.sqrt(2)
            sigmas.append(sigma_x)
            sigmas.append(sigma_y)
    outer_ii = jnp.einsum("ai,bi->iab",Q,Q.conj())
    for m in range(2**k-1):
        sigma_z = jnp.zeros([2**n,2**n],dtype=jnp.complex64)
        for i in range(m+1):
            sigma_z = sigma_z + outer_ii[i]/jnp.sqrt((m+1)*(m+2))
        sigma_z = sigma_z - (m+1)*outer_ii[m+1]/jnp.sqrt((m+1)*(m+2))
        sigmas.append(sigma_z)
    return jnp.stack(sigmas)

def max_eig(Q_real,Q_imag,n,k,p):
    Q = Q_real + 1j*Q_imag
    # n = int(jnp.log2(Q.shape[0]))
    # k = int(jnp.log2(Q.shape[1]))
    Pi = jnp.einsum('ip, jp -> ij', Q, Q.conj())
    Pi_til = jnp.einsum("ij,kl->ikjl",Pi,Pi.conj()).reshape([4**n,4**n])
    D = noise_super_operator(p)
    # print("generating N_til from D, n = ",n)
    N_til = jnp.einsum("ijkl->ikjl",tensor_power(D,n)).reshape([4**n,4**n])
    N_Pi = jnp.einsum("ij,j->i",N_til,Pi.reshape(4**n)).reshape([2**n,2**n])
    R_N_Pi_til = 0.5*(jnp.kron(N_Pi,jnp.eye(2**n))+jnp.kron(jnp.eye(2**n),N_Pi.conj()))
    R_inverse_til = jnp.linalg.pinv(R_N_Pi_til,rcond=1e-3, hermitian=True)
    M = jnp.einsum("ij,jk,kl,lm,mn->in",Pi_til,N_til,R_inverse_til,N_til,Pi_til,optimize=True)
    sigmas = generate_logical_hermitian_basis(Q,n,k)
    
    # 使用向量化操作计算J矩阵
    sigmas_flat = sigmas.reshape(4**k-1, 4**n)
    J = jnp.einsum("ia,ab,jb->ij", sigmas_flat.conj(), M, sigmas_flat)
    
    eigs = jnp.linalg.eigvalsh(J)
    return eigs.max()

if __name__ == "__main__":
    p = 0.1
    k = 1
    n = 5

    Q = trivial_Q(n)
    Q_real = jnp.real(Q)
    Q_imag = jnp.imag(Q)
    print(max_eig(Q_real,Q_imag,n,k,p))
