from PEC_base import *
from visualization import plot_cost_comparison, plot_survival_probability


def get_pauli(pauli_string):
    r""""
    从输入的形如"XYZIXI"的字符串出发，生成对应的pauli算符，shape=(2**n,2**n):
        _______
       |       |
    i--|   Si  |--j
       |       |
        -------
    """
    I2 = jnp.array([[1+0j,0],[0,1]])
    X = jnp.array([[0+0j,1],[1,0]])
    Y = jnp.array([[0,-1j],[1j,0]])
    Z = jnp.array([[1+0j,0],[0,-1]])
    letters = {'I':I2,'X':X,'Y':Y,'Z':Z}
    pauli_list = [letters[pauli] for pauli in pauli_string]
    if len(pauli_list) == 1:
        return pauli_list[0]
    while len(pauli_list) > 1:
        new_list = []
        for i in range(0,len(pauli_list),2):
            if i+1 < len(pauli_list):
                new_list.append(jnp.einsum("ij,kl->ikjl",pauli_list[i],pauli_list[i+1]).reshape([pauli_list[i].shape[0]*pauli_list[i+1].shape[0],pauli_list[i].shape[0]*pauli_list[i+1].shape[0]]))
            else:
                new_list.append(pauli_list[i])
        pauli_list = new_list
    return pauli_list[0]


def gen_projector(n,pauli_list):
    r"""
    从输入的一系列stabilizer出发，生成向量化之后的投影信道，shape=(2**n,2**n,2**n,2**n)，下标顺序[i,j,k,l]按照如图形式排列:
        _______
       |       |
    i--|   P   |--j
       |       |
        -------
        _______
       |       |
    k--|  P^*  |--l
       |       |
        -------
    """
    I = jnp.eye(2**n,dtype=pauli_list[0].dtype)
    if len(pauli_list) == 0:
        return jnp.einsum("ij,kl->ijkl",I,I)
    projector_list = []
    for stabilizer in pauli_list:
        new = (I+stabilizer)/2
        projector_list.append(jnp.einsum("ij,kl->ijkl",new,new.conj()))
    if len(pauli_list) == 1:
        return projector_list[0]
    else:
        result = projector_list[0]
        for i in range(1,len(projector_list)):
            result = jnp.einsum("ijkl,jalb->iakb",result,projector_list[i])
        return result

def gen_gamma(n,pauli_list):
    r"""
    生成逻辑最大混态gamma，也就是$( \prod_i (I+S_i)/2 ) / (2**k)$ ，返回的形状是(2**n,2**n)的矩阵:
        _______
       |       |
    i--| gamma |--j
       |       |
        -------
    """
    I = jnp.eye(2**n,dtype=pauli_list[0].dtype)
    if len(pauli_list) == 0:
        return I / (2**n)
    k = n - len(pauli_list) 
    if k == n-1 :
        return (I + pauli_list[0]) / (2**n)
    else:
        result = I + pauli_list[0]
        for i in range(1,n-k):
            result = jnp.einsum("ij,jk->ik",result,I+pauli_list[i])
        return result / (2**n)

def calculate_trace(n,channel,rho):
    r"""
    计算任何一个量子态在经过CP channel后存活的概率:
        _______       _______
       |       |     |       |
    i--|       |--j--|  rho  |--l--
    |  |       |     |       |     |
    |  |   ~   |      -------      |
    |  |   C   |                   |
    |  |       |                   |
    i--|       |--l----------------
       |       |
        -------
    """
    return jnp.einsum("ijil,jl->",channel,rho).real


def composite(n,projector,channel):
    r"""
    给出一个CPTP channel与投影测量复合而成的CP map:
        _______       _______
       |       |     |       |
    i--|       |--j--|       |--a
       |       |     |       |
       |   ~   |     |   ~   |
       |   P   |     |   C   |
       |       |     |       |
    k--|       |--l--|       |--b
       |       |     |       |
        -------       -------
    """
    return jnp.einsum("ijkl,jalb->iakb",projector,channel)


def post_select(n,projector,channel,pauli_list):
    CP_map = composite(n,projector,channel)
    
    gamma = gen_gamma(n,pauli_list)
    probability = calculate_trace(n,CP_map,gamma)
    reduced_N = CP_map/probability

    return probability , reduced_N



def get_transfer_vector(n,k,reduced_N,projector):
    r"""
    计算reduced_N作用在所有pauli算符上产生的效果:
                                  i(0)
                                   |
               --------------------|--------------------
              |                                         |
           ___|___       _______       _______       ___|___
          |       |     |       |     |       |     |       |
     -----|1  P  2|--0--|       |--1--|       |--4--|3  P  4|-----
    |     |       |     |       |     |       |     |       |     |
    |      -------      |   ~   |     |   ~   |      -------      |
    |                   |   N   |     |   Pi  |                   |
    |                   |       |     |       |                   |
     ----------------2--|       |--3--|       |--5----------------
                        |       |     |       |     
                         -------       -------      
    """
    I2 = jnp.array([[1+0j,0],[0,1]])
    X = jnp.array([[0+0j,1],[1,0]])
    Y = jnp.array([[0,-1j],[1j,0]])
    Z = jnp.array([[1+0j,0],[0,-1]])  


    II = jnp.einsum("ij,kl->ijkl",I2,I2)
    XX = jnp.einsum("ij,kl->ijkl",X,X)
    YY = jnp.einsum("ij,kl->ijkl",Y,Y)
    ZZ = jnp.einsum("ij,kl->ijkl",Z,Z)
    
    N = jnp.einsum("ijkl,jakb->iakb",reduced_N,projector).reshape([2]*(4*n))
    permute_index = []
    for i in range(n):
        permute_index.append(i)
        permute_index.append(i+n)
        permute_index.append(i+2*n)
        permute_index.append(i+3*n)
    N = jnp.transpose(N,permute_index)
    local_operator = jnp.stack([II,XX,YY,ZZ],axis=0)
    for _ in range(n):
        N = jnp.tensordot(N,local_operator,axes=([0,1,2,3],[2,3,1,4]))
    transfer_vector = N.ravel().real / (2**k)
    return transfer_vector


def get_coefficient_vector(n,k,transfer_vector):
    r"""
    输入pauli channel的transfer vector，是shape=(4**n,)的一个实向量，代表channel对pauli算符的缩放行为: reduced_N(P_i) = lambda_i P_i，其中只有4**k个在逻辑子空间内的lambda_i非0
    输出为了在逻辑层面实现这个pauli channel所需要的各个pauli kraus算符前的系数向量，shape=(4**k,)
     
    原理上:
    reduced_N(P_j) = sum_i coefficient_i P_i P_j P_i = sum_i coefficient_i (-1)^{<Pi,Pj>} P_j = lambda_j P_j
    也就是:
    sum_i coefficient_i (-1)^{<Pi,Pj>} = lambda_j
    上式是一个线性方程组，我们知道lambda向量，想解coefficient向量，可以给出这个矩阵的逆:
    sum_j (-1)^{<Pi,Pj>} * (-1)^{<Pj,Pk>} = sum_j (-1)^{<Pj,Pi> + <Pj,Pk>} = sum_j (-1)^{<Pj,Pi@Pk>} = (4**k) * delta_{i,k}
    因此可以反解出来:    
    coefficient_i = (1/4**k) sum_j(-1)^{<Pi,Pj>} lambda_j
    
    为了正确得到WHT矩阵，需要首先考虑算符定义的顺序，根据PEC_base中的定义，算符按照例如: "III","IIX","IIY","IIZ","IXI","IXX","IXY","IXZ",...的顺序排列，我们需要正确给出被stabilizer筛选之后的pauli算符之间的对易关系矩阵，而这只需要mask整体n qubit pauli算符WHT矩阵即可得到。
    
    我们猜测整体满足的对易关系是如下矩阵的n次张量积:
    
    1  1  1  1
    1  1 -1 -1
    1 -1  1 -1
    1 -1 -1  1
    
    经过检验的确没有问题。
    
    我们首先需要识别出lambda_i向量占4**n维向量的哪些元素，得到对应的mask向量，并判断是不是4**k维
    
    我们发现这里会出现由于多一个stabilizer带来的简并
    """
    magnitudes = jnp.abs(transfer_vector)
    _, indices = jax.lax.top_k(magnitudes, 2**(n+k))
    indices = jnp.sort(indices)
    v_sub = transfer_vector[indices]
    eps = 1e-7
    v_sub = jnp.where(jnp.abs(v_sub) > eps, 1.0 / v_sub, 0.0)
    
    single_WHT = jnp.array([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]],dtype=transfer_vector.dtype)
    WHT = single_WHT
    for i in range(n-1):
        WHT = jnp.kron(WHT,single_WHT)
    w_sub = WHT[indices][:, indices]
    coefficient_vector = jnp.einsum("ij,j->i", w_sub, v_sub) / (4**k)
    return coefficient_vector / (4**(n-k))

from functools import partial

@partial(jax.jit, static_argnames=('n', 'k', 'L', 'stabilizer_strings'))
def subspace_total_PEC(n,k,p,std,L,key,stabilizer_strings):
    pauli_list = [get_pauli(stabilizer_string) for stabilizer_string in stabilizer_strings]
    projector = gen_projector(n,pauli_list)
    error_matrix = gen_error(p,std,n,key)
    layered_error_matrix = gen_layered_error(error_matrix,L)
    # print("layered_error_matrix[0,0] = ",layered_error_matrix[0,0])
    N = tensor_single_pauli_noise(n,layered_error_matrix)
    probability, reduced_N = post_select(n,projector,N,pauli_list)


    transfer_vector = get_transfer_vector(n,k,reduced_N,projector)
    # print("transfer_vector = ",transfer_vector)
    
    coefficient_vector = get_coefficient_vector(n,k,transfer_vector)
    PEC_sample = (jnp.abs(coefficient_vector).sum())**2
    
    return probability, PEC_sample


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n = 4
    std = 0
    
    stabilizer_string = ["X"*n, "Z"*n]
    k = n - len(stabilizer_string)

    L_list = np.arange(1,3002,1000, dtype=int)
    # p_list = jnp.linspace(0, 0.001, 10)
    # p_list = np.array([0.0001,0.001,0.01])
    p_list = np.array([0.0003])

    probs_matrix = np.zeros((len(p_list), len(L_list)))
    pec_samples_matrix = np.zeros((len(p_list), len(L_list)))
    total_samples_matrix = np.zeros((len(p_list), len(L_list)))

    print(f"Running simulation for L in ")
    print(L_list)
    print("and p in ")
    print(p_list)
    
    for i, p in enumerate(p_list):
        for j, L in enumerate(L_list):
            probability, PEC_sample = subspace_total_PEC(n, k, p, std, int(L), key, tuple(stabilizer_string))
            total_sample = PEC_sample / probability
            
            probs_matrix[i, j] = probability
            pec_samples_matrix[i, j] = PEC_sample
            total_samples_matrix[i, j] = total_sample
            # print("----------------")
            # print("L = ",L)
            # print("probability = ",probability)
            # print("PEC_sample = ",np.log10(PEC_sample))
            # print("log(total_sample) = ",np.log10(total_sample))
            # print("----------------")

    # combined = jnp.column_stack([probs_matrix[0], np.log10(pec_samples_matrix[0]), np.log10(total_samples_matrix[0])])
    # np.savetxt('output.csv', combined, delimiter=',', fmt='%s')
    plot_survival_probability(L_list, p_list, probs_matrix)
    plot_cost_comparison(L_list, p_list, pec_samples_matrix, total_samples_matrix)
