from layered_subspace_PEC import *
from visualization import plot_all, save_results_to_csv


# if __name__ == "__main__":
#     key = jax.random.PRNGKey(0)
#     n = 6
#     std = 0
#     p = 0.0003
    
#     stabilizer_string = ["X"*n, "Z"*n]
#     k = n - 2
    
#     pauli_list = [get_pauli(s) for s in stabilizer_string]
#     projector = gen_projector(n,pauli_list)
    
#     error_matrix = gen_error(p,std,n,key)
    


#     L_list = np.arange(1,3003,100)
#     L_check_list = np.array([1,10,30,50,70,100,200,300,400,500,600])
#     # L_check_list = np.array([700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000])

#     p_matrix = np.zeros((len(L_check_list),len(L_list)))
#     pec_matrix = np.zeros((len(L_check_list),len(L_list)))
#     total_matrix = np.zeros((len(L_check_list),len(L_list)))


#     for i, L_check in enumerate(L_check_list):
#         print("starting L_check = ",L_check)
#         layered_error_matrix1 = gen_layered_error(error_matrix,L_check)
#         N1 = tensor_single_pauli_noise(n,layered_error_matrix1)
#         probability1, reduced_N1 = post_select(n,projector,N1,pauli_list)
#         transfer_vector1 = get_transfer_vector(n,k,reduced_N1,projector)
#         for j, L in enumerate(L_list):
#             print("         processing L = ",L)
#             layered_error_matrix2 = gen_layered_error(error_matrix,L%L_check)
#             N2 = tensor_single_pauli_noise(n,layered_error_matrix2)
#             probability2, reduced_N2 = post_select(n,projector,N2,pauli_list)
#             transfer_vector2 = get_transfer_vector(n,k,reduced_N2,projector)
#             transfer_vector = (transfer_vector1**(L//L_check)) * (transfer_vector2)
#             coefficient_vector = get_coefficient_vector(n,k,transfer_vector)
#             PEC_sample = (jnp.abs(coefficient_vector).sum())**2
#             probability = (probability1**(L//L_check)) * (probability2)
            
#             p_matrix[i,j] = probability
#             pec_matrix[i,j] = PEC_sample
#             total_matrix[i,j] = PEC_sample / probability
    





# 封装并JIT编译关键计算步骤
@partial(jax.jit, static_argnames=('n',))
def compute_noise_and_postselect(n, layered_error_matrix, projector, pauli_list):
    """计算噪声矩阵并进行后选择"""
    N = tensor_single_pauli_noise(n, layered_error_matrix)
    probability, reduced_N = post_select(n, projector, N, pauli_list)
    return probability, reduced_N


@partial(jax.jit, static_argnames=('n', 'k'))
def compute_transfer_and_coefficient(n, k, reduced_N, projector):
    """计算传输向量和系数向量"""
    transfer_vector = get_transfer_vector(n, k, reduced_N, projector)
    return transfer_vector


@partial(jax.jit, static_argnames=('n', 'k'))
def compute_pec_sample(n, k, transfer_vector, projector):
    """计算PEC采样值"""
    coefficient_vector = get_coefficient_vector(n, k, transfer_vector)
    PEC_sample = (jnp.abs(coefficient_vector).sum())**2
    return PEC_sample


@partial(jax.jit, static_argnames=())
def compute_combined_transfer_and_pec(transfer_vector1, transfer_vector2, 
                                       probability1, probability2, 
                                       L_quotient):
    """组合传输向量并计算最终概率"""
    transfer_vector = (transfer_vector1**L_quotient) * transfer_vector2
    probability = (probability1**L_quotient) * probability2
    return transfer_vector, probability


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n = 6
    std = 0
    p = 0.0003
    
    stabilizer_string = ["X"*n, "Z"*n]
    k = n - 2
    
    pauli_list = [get_pauli(s) for s in stabilizer_string]
    projector = gen_projector(n, pauli_list)
    
    error_matrix = gen_error(p, std, n, key)
    
    L_list = np.array([1001,2001,3001])
    L_check_list = np.array([10,100,500])
    # L_list = np.arange(1, 3003, 100)
    # L_check_list = np.array([1, 10, 30, 50, 70, 100, 200, 300, 400, 500, 600])
    # L_check_list = np.array([700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000])
    

    p_matrix = np.zeros((len(L_check_list), len(L_list)))
    pec_matrix = np.zeros((len(L_check_list), len(L_list)))
    total_matrix = np.zeros((len(L_check_list), len(L_list)))
    
    # 预计算并缓存外层循环的结果
    cache_prob1 = {}
    cache_transfer1 = {}
    
    for i, L_check in enumerate(L_check_list):
        print("starting L_check =", L_check)
        
        # 计算外层循环的固定值（每个L_check只计算一次）
        if L_check not in cache_prob1:
            layered_error_matrix1 = gen_layered_error(error_matrix, L_check)
            probability1, reduced_N1 = compute_noise_and_postselect(
                n, layered_error_matrix1, projector, pauli_list
            )
            transfer_vector1 = compute_transfer_and_coefficient(n, k, reduced_N1, projector)
            
            cache_prob1[L_check] = probability1
            cache_transfer1[L_check] = transfer_vector1
        else:
            probability1 = cache_prob1[L_check]
            transfer_vector1 = cache_transfer1[L_check]
        
        for j, L in enumerate(L_list):
            print("         processing L =", L)
            
            L_remainder = L % L_check
            L_quotient = L // L_check
            
            # 计算L_remainder对应的值
            layered_error_matrix2 = gen_layered_error(error_matrix, L_remainder)
            probability2, reduced_N2 = compute_noise_and_postselect(
                n, layered_error_matrix2, projector, pauli_list
            )
            transfer_vector2 = compute_transfer_and_coefficient(n, k, reduced_N2, projector)
            
            # 组合计算
            transfer_vector, probability = compute_combined_transfer_and_pec(
                transfer_vector1, transfer_vector2, 
                probability1, probability2,
                L_quotient
            )
            
            # 计算PEC样本
            PEC_sample = compute_pec_sample(n, k, transfer_vector, projector)
            
            p_matrix[i, j] = probability
            pec_matrix[i, j] = PEC_sample
            total_matrix[i, j] = PEC_sample / probability
            

    plot_all(L_list, L_check_list, p_matrix, pec_matrix, total_matrix, prefix='')
    save_results_to_csv(L_list, L_check_list, p_matrix, pec_matrix, 
                    total_matrix, save_path='results.csv')
