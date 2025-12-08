import numpy as np
import jax.numpy as jnp
import jax
import monitor
from matplotlib import pyplot as plt

def single_D(px,py,pz):
    r"""
    return一个2*2*2*2张量，实际上是向量化的单个qubit上的channel的某种表示，此时下标的顺序是[i,j,k,l]如图:
        ______
       |      |
    i--|  P   |--j
       |      |
        ------
        ______
       |      |
    k--|  P^* |--l
       |      |
        ------
    这样做是为了便于之后进行tensor运算
    
    为了将一个纯态向量psi写为一个密度矩阵rho_psi，进而写为一个超向量vec_psi，只需要:
        rho_psi = einsum("i,j->ij",psi,psi.conj())
        vec_psi = rho_psi.reshape(rho_psi.shape[0]*2)

    为了decode一个超向量形式的密度矩阵，只需要进行
        rho.reshape([ rho.shape[0]//2 , rho.shape[0]//2 ])
    """
    I2 = jnp.array([[1+0j,0],[0,1]])
    X = jnp.array([[0+0j,1],[1,0]])
    Y = jnp.array([[0,-1j],[1j,0]])
    Z = jnp.array([[1+0j,0],[0,-1]])
    return (1-px-py-pz)*jnp.einsum("ij,kl->ijkl",I2,I2) + px*jnp.einsum("ij,kl->ijkl",X,X) \
        + py*jnp.einsum("ij,kl->ijkl",Y,Y.conj()) + pz*jnp.einsum("ij,kl->ijkl",Z,Z)

def tensor(C1,C2):
    r"""
    两个channel进行tensor可以使用 einsum("ijkl,abcd->iajbkcld",A,B).reshape([Ai*Bi , Aj*Bj , Ak*Bk , Al*Bl])，根据numpy的下标规则，这样恰好可以得到tensor之后的矩阵。这样直积之后得到的仍然是[i,j,k,l]如图顺序的超矩阵:
        ______
       |      |
    i--|  P   |--j
       |      |
        ------
        ______
       |      |
    k--|  P^* |--l
       |      |
        ------
    而为了能够将channel复合作为矩阵乘法进行运算，必须要再执行一个下标的重排 einsum("ijkl->ikjl",X).reshape([Xi*Xk , Xj*Xl]), 这样得到的超矩阵之间的乘法才对应channel之间的复合:
        C1_C2 = einsum("ij,jk->ik",C1,C2)
        ______       ______
       |      |     |      |
    i--|  C1  |--j--|  C2  |--k
       |      |     |      |
        ------       ------
    """
    i,j,k,l = C1.shape
    a,b,c,d = C2.shape
    return jnp.einsum("ijkl,abcd->iajbkcld",C1,C2).reshape([i*a,j*b,k*c,d*l])

def gen_error(p,std,n,key):
    error_matrix = jax.random.uniform(key, shape=(n,3), minval=p-std/2, maxval=p+std/2)
    return error_matrix

def gen_layered_error(error_matrix,L):
    pyz = (1-2*(error_matrix[:,1]+error_matrix[:,2]))**L
    pxy = (1-2*(error_matrix[:,0]+error_matrix[:,1]))**L
    pxz = (1-2*(error_matrix[:,0]+error_matrix[:,2]))**L
    px = (pyz - pxy - pxz + 1) / 4
    py = (pxz - pxy - pyz + 1) / 4
    pz = (pxy - pxz - pyz + 1) / 4

    layered_error_matrix = jnp.stack([px , py , pz],axis=1)
    return layered_error_matrix


def tensor_single_pauli_noise(n,error_matrix):
    r"""
    利用二分法减少tensor过程的计算量，得到的仍然是[i,j,k,l]如图顺序的超矩阵:
        ______
       |      |
    i--|  P   |--j
       |      |
        ------
        ______
       |      |
    k--|  P^* |--l
       |      |
        ------
    """
    def tensor_power(index_start,index_end):
        if index_start == index_end:
            return single_D(error_matrix[index_start][0],error_matrix[index_start][1],error_matrix[index_start][2])
        elif index_end == index_start+1:
            return tensor(single_D(error_matrix[index_start][0],error_matrix[index_start][1],error_matrix[index_start][2]),single_D(error_matrix[index_end][0],error_matrix[index_end][1],error_matrix[index_end][2]))
        else:
            index_middle = index_start + (index_end-index_start)//2
            return tensor(tensor_power(index_start,index_middle),tensor_power(index_middle+1,index_end))
    return tensor_power(0,n-1)

def single_all_pauli():
    r"""
    单比特上的所有可能pauli算符，指标顺序为[m,i,j,k,l]如图:
    
    m=0(I),1(X),2(Y),3(Z)
           |
        ___|__
       |      |
    i--|  P   |--j
       |      |
        ------
        ______
       |      |
    k--|  P^* |--l
       |      |
        ------
    shape=(4,2,2,2,2)
    """
    I2 = jnp.array([[1+0j,0],[0,1]])
    X = jnp.array([[0+0j,1],[1,0]])
    Y = jnp.array([[0,-1j],[1j,0]])
    Z = jnp.array([[1+0j,0],[0,-1]])

    return jnp.stack([jnp.einsum("ij,kl->ijkl",I2,I2),jnp.einsum("ij,kl->ijkl",X,X),jnp.einsum("ij,kl->ijkl",Y,Y.conj()),jnp.einsum("ij,kl->ijkl",Z,Z)],axis=0)

def all_pauli(n):
    r"""
    利用二分法构造一种包含了n qubit所有可能pauli算符的张量，指标顺序为[m,i,j,k,l]如图:

           m
           |
        ___|__
       |      |
    i--|  P   |--j
       |      |
        ------
        ______
       |      |
    k--|  P^* |--l
       |      |
        ------
    shape=(4**n,2**n,2**n,2**n,2**n,2**n)
    """
    if n==1:
        return single_all_pauli()
    elif n%2 == 0:
        return jnp.einsum("qwert,yuiop->qywueirotp",all_pauli(n//2),all_pauli(n//2)).reshape([4**n,2**n,2**n,2**n,2**n])
    elif n%2 == 1:
        return jnp.einsum("qwert,yuiop,asdfg->qyawuseidroftpg",all_pauli(n//2),all_pauli(n//2),single_all_pauli(),optimize=True).reshape([4**n,2**n,2**n,2**n,2**n])

def PEC_sample_cost(n,N_inv):
    r"""
    计算sample cost的第一步是计算各个pauli分量的准概率向量:
           m                                              m                                               
           |                                              |                                               
        ___|__       ______                            ___|__       ______                            m
       |      |     |      |                          |      |     |      |                         __|___
    i--|  Pm  |--j--|      |--i                    i--|  Pm  |--j--|   Q  |--i                     |      |
       |      |     |      |                          |      |     |      |                        |      |
        ------      | ~    |      =   \sum_Q  c_Q      ------       ------        =   4^n *  c_m   |  1   |
        ______      | N^-1 |                           ______       ______                         |      |
       |      |     |      |                          |      |     |      |                         ------ 
    k--| Pm^* |--l--|      |--k                    k--| Pm^* |--l--|  Q^* |--k                            
       |      |     |      |                          |      |     |      |                               
        ------       ------                            ------       ------                                

    """
    quasi_error_vector = (1/(4**n))*jnp.einsum("mijkl,jilk->m",all_pauli(n),N_inv)
    return (jnp.abs(quasi_error_vector).sum().real)**2


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n = 4
    p = 0.01
    std = 0


    # p_list = jnp.linspace(0,0.01,10)
    p_list = jnp.array([0.0001])
    L_list = jnp.arange(1,1000,100)
    plt.figure()
    for p in p_list:
        error_matrix = gen_error(p,std,n,key)
        result = []
        for L in L_list:
            layered_error_matrix = gen_layered_error(error_matrix,L)
            NL = tensor_single_pauli_noise(n,layered_error_matrix)
            NL_matrix = jnp.einsum("ijkl->ikjl",NL).reshape([4**n,4**n])
            NL_inv_matrix = jnp.linalg.pinv(NL_matrix,rtol=1e-10)
            NL_inv = jnp.einsum("ikjl->ijkl",NL_inv_matrix.reshape([2**n,2**n,2**n,2**n]))
            result.append(jnp.log(PEC_sample_cost(n,NL_inv)))
            # result.append(sample_cost(n,NL_inv))
        plt.plot(L_list,result,label="p="+str(p))
    plt.title("p = "+str(p))
    plt.legend()
    plt.show()
