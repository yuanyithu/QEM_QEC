import numpy as np
import matplotlib.pyplot as plt

def gen_vec(positions,name,n):
    """
    生成长度为2n的pauli算符辛表示，先X后Z
    """
    result = np.zeros(2*n,dtype=np.int16)
    if len(positions) == 1:
        i = int(positions[0])
        if name == "X":
            result[i] = 1
            return result
        elif name == "Y":
            result[i] = 1
            result[i+n] = 1
            return result
        elif name == "Z":
            result[i+n] = 1
            return result
    elif len(positions) == 2:
        i,j = positions
        return gen_vec([i],name[0],n) + gen_vec([j],name[1],n)

# gate_info_1 = ["CNOT",[1,2]]
# gate_info_2 = ["X",[0]]
# gate_info_3 = ["ZY",[0,2]]

# def single_layer_propagate(gate_info,error,n):
#     if gate_info[0] == "CNOT":
#         i,j = gate_info[1]
#         error[j] ^= error[i]
#         error[i+n] ^= error[j+n]
#         return error


def init_gate_info(n):
    gate_info_list1 = []
    for i in range(0,n,2):
        gate_info_list1.append(["CNOT",[i,(i+1)%n]])

    gate_info_list2 = []
    for i in range(1,n,2):
        gate_info_list2.append(["CNOT",[i,(i+1)%n]])

    return gate_info_list1, gate_info_list2

def propagate(n,T,p1,p2):
    error_list = []
    error_parameter_list = []
    for i in range(n):
        for error_type in ["X","Y","Z"]:
            error_parameter_list.append(p1) # 单比特噪声参数
            error_list.append(gen_vec([i],error_type,n)) # 3n个单比特噪声
        for error_type in [a+b for a in ["X","Y","Z"] for b in ["X","Y","Z"]]:
            error_parameter_list.append(p2) # 双比特噪声参数
            error_list.append(gen_vec([i,(i+1)%n],error_type,n)) # 9n个双比特噪声

    gate_info_list_set = init_gate_info(n)
    cycle = len(gate_info_list_set)
    for l in range(T-1):
        # gate_info_list = gate_info_list_set[l%cycle]
        # # 传播所有已有噪声
        # for i,error in enumerate(error_list):
        #     for gate_info in gate_info_list:
        #         error_list[i] = single_layer_propagate(gate_info,error,n)
        # 增添所有新的噪声
        for i in range(n):
            for error_type in ["X","Y","Z"]:
                error_parameter_list.append(p1) # 单比特噪声参数
                error_list.append(gen_vec([i],error_type,n)) # 3n个单比特噪声
            for error_type in [a+b for a in ["X","Y","Z"] for b in ["X","Y","Z"]]:
                error_parameter_list.append(p2) # 双比特噪声参数
                error_list.append(gen_vec([i,(i+1)%n],error_type,n)) # 9n个双比特噪声

    return error_list,error_parameter_list

def gen_stabilizer_list(n):
    all_X = np.concatenate([np.ones(n, dtype=np.int16), np.zeros(n, dtype=np.int16)])
    all_Z = np.concatenate([np.zeros(n, dtype=np.int16), np.ones(n, dtype=np.int16)])
    return [all_X , all_Z]


def error_detect(n,error_list,stabilizer_list,error_parameter_list):
    fail_weight = 0
    success_weight = 0
    for i,error in enumerate(error_list):
        check1 = (error[:n] @ stabilizer_list[0][n:] + error[n:] @ stabilizer_list[0][:n]) % 2
        check2 = (error[:n] @ stabilizer_list[1][n:] + error[n:] @ stabilizer_list[1][:n]) % 2
        if (check1 == 1) or (check2 == 1): # 不通过
            fail_weight += error_parameter_list[i]
        else: # 通过
            success_weight += error_parameter_list[i]
    return fail_weight, success_weight

def sample_cost(n,p1,p2,T):
    # print("n * p * T ~ ",n*T*p*9)
    error_list, error_parameter_list = propagate(n,T,p1,p2)
    stabilizer_list = gen_stabilizer_list(n)
    fail_weight, success_weight = error_detect(n,error_list,stabilizer_list,error_parameter_list)
    
    # print("fail_terms = ",fail_terms)
    # print("error_terms = ",error_terms)

    p_success = 1 - fail_weight
    gamma = 1 + 2 * success_weight / p_success
    
    sample = gamma**2 / p_success
    
    return sample,fail_weight,success_weight



if __name__ == "__main__":
    n = 1000
    p1 = 1e-6
    p2 = p1**2
    L = 3000
    
    T_list = np.arange(1,50,2)
    # T_list = [1,5,10,50,100]
    total_sample_list = []
    fail_weight_list = []
    success_weight_list = []
    for T in T_list:
        sample_1, fail_weight_1, success_weight_1 = sample_cost(n,p1,p2,T)
        if L%T == 0:
            sample_2 = 1
            fail_weight_list.append(fail_weight_1)
            success_weight_list.append(success_weight_1)
        else:
            sample_2, fail_weight_2, success_weight_2 = sample_cost(n,p1,p2,L%T)
            fail_weight_list.append(max(fail_weight_1,fail_weight_2))
            success_weight_list.append(max(success_weight_1,success_weight_2))
        total_sample = (sample_1**(L//T)) * sample_2
        # print("p1 = ",p1,"  T = ",T,"  log10(sample) = ",np.log10(total_sample))
        total_sample_list.append(total_sample)
        
    
    total_sample_list = np.array(total_sample_list)
    fail_weight_list = np.array(fail_weight_list)
    success_weight_list = np.array(success_weight_list)
    plt.plot(T_list,np.log10(total_sample_list))
    plt.title("log10(total sample)")
    plt.show()
    plt.plot(T_list,fail_weight_list)
    plt.title("fail_weight")
    plt.show()
    plt.plot(T_list,success_weight_list)
    plt.title("success_weight")
    plt.show()