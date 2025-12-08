from subspace_PEC import *
from visualization import plot_all, save_results_to_csv



@partial(jax.jit, static_argnames=('n', 'k', 'stabilizer_strings'))
def subspace_total_PEC(n,k,p,std,L,key,stabilizer_strings):
    pauli_list = [get_pauli(stabilizer_string) for stabilizer_string in stabilizer_strings]
    projector = gen_projector(n,pauli_list)
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
    n = 4
    std = 0

    stabilizer_string = ["X"*n, "Z"*n]
    k = n - len(stabilizer_string)


    # L_check_list = np.arange(1,2000,100,dtype=int)
    L_check_list = np.array([1,5,10,50,100])
    # L_check_list = np.array([700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000])
    # L_list = np.arange(1,3003,100, dtype=int)
    L_list = np.array([3000])
    # p_list = jnp.linspace(0, 0.001, 2)
    # p_list = np.array([0.0001,0.001,0.01])
    p_list = np.array([1e-4])

    probs_matrix = np.zeros((len(L_check_list),len(p_list), len(L_list)))
    pec_samples_matrix = np.zeros((len(L_check_list),len(p_list), len(L_list)))
    total_samples_matrix = np.zeros((len(L_check_list),len(p_list), len(L_list)))

    # print("Running simulation for L in ")
    # print(L_list)
    # print("and p in ")
    # print(p_list)
    # print("and L_check in ")
    # print(L_check_list)
    # print("--------------------")


    for i, L_check in enumerate(L_check_list):
        
        for j, p in enumerate(p_list):
            
            for l, L in enumerate(L_list):
                print("L = ",L,end=" , ")
                print("L_check = ",L_check,end=" , ")
                print("p = ",p,end=" : ")
                probability1, PEC_sample1 = subspace_total_PEC(n, k, p, std, int(L_check), key, tuple(stabilizer_string))
                probability2, PEC_sample2 = subspace_total_PEC(n, k, p, std, int(L%L_check), key, tuple(stabilizer_string))
                probability = (probability1**(L//L_check))*probability2
                PEC_sample = (PEC_sample1**(L//L_check))*PEC_sample2
                total_sample = PEC_sample / probability
                print("T total sample = ",PEC_sample1 / probability1,end=" , ")
                print("log10(total sample) = ",np.log10(total_sample))

                # probs_matrix[i, j, l] = probability
                # pec_samples_matrix[i, j, l] = PEC_sample
                # total_samples_matrix[i, j, l] = total_sample
                # print("----------------")
                # print("L = ",L)
                # print("probability = ",probability)
                # print("PEC_sample = ",PEC_sample)
                # print("total_sample = ",total_sample)
                # print("----------------")


    # plot_all(L_list, L_check_list, probs_matrix, pec_samples_matrix, total_samples_matrix, prefix='layered_')
    # save_results_to_csv(L_list, L_check_list, probs_matrix, pec_samples_matrix, 
                    #    total_samples_matrix, save_path='layered_PEC_results.csv')
