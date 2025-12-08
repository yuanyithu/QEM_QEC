from single_round_jax import *
from matplotlib import pyplot as plt

def generate_Q(n):
    Q_zeros = jnp.zeros([2**n,2],dtype=jnp.complex64)
    Q = (Q_zeros.at[0,0].set(1)).at[2**n-1,1].set(1)
    return Q

def all_eig(Q_real,Q_imag,n,k,p):
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
    return eigs

if __name__ == "__main__":
    p = 0.1
    k = 1
    n_list = [1,2,3,4,5,6]
    im = plt.figure()

    for n in n_list:
        Q = generate_Q(n)
        Q_real = jnp.real(Q)
        Q_imag = jnp.imag(Q)
        eigs = all_eig(Q_real,Q_imag,n,k,p)
        for eig in eigs:
            plt.scatter(n,eig)

    plt.show()