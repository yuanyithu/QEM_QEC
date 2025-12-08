from layered_total_subspace_PEC import *


def gen_projector_pro(n,pauli_list,signs):
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
    signs = [1,-1]代表第一个stabilizer为+1而第二个stabilizer为-1的子空间
    """
    I = jnp.eye(2**n,dtype=pauli_list[0].dtype)
    if len(pauli_list) == 0:
        return jnp.einsum("ij,kl->ijkl",I,I)
    projector_list = []
    for i,stabilizer in enumerate(pauli_list):
        new = (I+signs[i]*stabilizer)/2
        projector_list.append(jnp.einsum("ij,kl->ijkl",new,new.conj()))
    if len(pauli_list) == 1:
        return projector_list[0]
    else:
        result = projector_list[0]
        for i in range(1,len(projector_list)):
            result = jnp.einsum("ijkl,jalb->iakb",result,projector_list[i])
        return result

def subspace_total_PEC(n,k,p,std,L,key,stabilizer_strings,signs):
    pauli_list = [get_pauli(stabilizer_string) for stabilizer_string in stabilizer_strings]
    projector = gen_projector_pro(n,pauli_list,signs)
    error_matrix = gen_error(p,std,n,key)

    layered_error_matrix = gen_layered_error(error_matrix,L)
    N = tensor_single_pauli_noise(n,layered_error_matrix)
    probability, reduced_N = post_select(n,projector,N,pauli_list)

    transfer_vector = get_transfer_vector(n,k,reduced_N,projector)

    coefficient_vector = get_coefficient_vector(n,k,transfer_vector)
    PEC_sample = (jnp.abs(coefficient_vector).sum())**2

    return probability, PEC_sample



if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n = 6
    std = 0

    stabilizer_string = ["X"*n, "Z"*n]
    k = n - len(stabilizer_string)

    L_check = 100
    L = 3001
    p = 0.0003
    
    probability1, PEC_sample1 = subspace_total_PEC(n, k, p, std, int(L_check), key, tuple(stabilizer_string),(1,1))
    probability2, PEC_sample2 = subspace_total_PEC(n, k, p, std, int(L_check), key, tuple(stabilizer_string),(1,-1))
    probability3, PEC_sample3 = subspace_total_PEC(n, k, p, std, int(L_check), key, tuple(stabilizer_string),(-1,1))
    probability4, PEC_sample4 = subspace_total_PEC(n, k, p, std, int(L_check), key, tuple(stabilizer_string),(-1,-1))
    print(probability1)
    print(probability2)
    print(probability3)
    print(probability4)
    


    # # L_check_list = np.arange(1,2000,100,dtype=int)
    # # L_check_list = np.array([1,10,30,50,70,100,200,300,400,500,600])
    # # L_check_list = np.array([700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000])
    # # L_list = np.arange(1,3003,100, dtype=int)
    # # p_list = jnp.linspace(0, 0.001, 2)
    # # p_list = np.array([0.0001,0.001,0.01])
    # p_list = np.array([0.0003])

    # probs_matrix = np.zeros((len(L_check_list),len(p_list), len(L_list)))
    # pec_samples_matrix = np.zeros((len(L_check_list),len(p_list), len(L_list)))
    # total_samples_matrix = np.zeros((len(L_check_list),len(p_list), len(L_list)))

    # print("Running simulation for L in ")
    # print(L_list)
    # print("and p in ")
    # print(p_list)
    # print("and L_check in ")
    # print(L_check_list)
    # print("--------------------")


    # for i, L_check in enumerate(L_check_list):
    #     print("starting L_check = ",L_check)
    #     for j, p in enumerate(p_list):
    #         for l, L in enumerate(L_list):
    #             print("         processing L = ",L)
    #             probability1, PEC_sample1 = subspace_total_PEC(n, k, p, std, int(L_check), key, tuple(stabilizer_string))
    #             probability2, PEC_sample2 = subspace_total_PEC(n, k, p, std, int(L%L_check), key, tuple(stabilizer_string))
    #             probability = (probability1**(L//L_check))*probability2
    #             PEC_sample = (PEC_sample1**(L//L_check))*PEC_sample2
    #             total_sample = PEC_sample / probability

    #             probs_matrix[i, j, l] = probability
    #             pec_samples_matrix[i, j, l] = PEC_sample
    #             total_samples_matrix[i, j, l] = total_sample
    #             # print("----------------")
    #             # print("L = ",L)
    #             # print("probability = ",probability)
    #             # print("PEC_sample = ",PEC_sample)
    #             # print("total_sample = ",total_sample)
    #             # print("----------------")


    # plot_all(L_list, L_check_list, probs_matrix, pec_samples_matrix, total_samples_matrix, prefix='layered_')
    # save_results_to_csv(L_list, L_check_list, probs_matrix, pec_samples_matrix, 
    #                    total_samples_matrix, save_path='layered_PEC_results.csv')
