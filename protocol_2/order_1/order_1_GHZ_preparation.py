"""

0------*------.-----.------*------.-----.------*------.-----.------*------.-----.------*------.-----.------*------.-----.------*------.-----.------*------
       *      |     |      *      |     |      *      |     |      *      |     |      *      |     |      *      |     |      *      |     |      *      
1------*------X--X--|------*------X--X--|------*------X--X--|------*------X--X--|------*------X--X--|------*------X--X--|------*------X--X--|------*------
       *         |  |      *         |  |      *         |  |      *         |  |      *         |  |      *         |  |      *         |  |      *      
2------*------.--.--|------*---------|--|------*---------|--|------*---------|--|------*---------|--|------*---------|--|------*---------|--|------*------
       *      |     |      *         |  |      *         |  |      *         |  |      *         |  |      *         |  |      *         |  |      *      
3------*------X-----X------*------.--.--|------*---------|--|------*---------|--|------*---------|--|------*---------|--|------*---------|--|------*------
       *                   *      |     |      *         |  |      *         |  |      *         |  |      *         |  |      *         |  |      *      
4------*-------------------*------X-----X------*------.--.--|------*---------|--|------*---------|--|------*---------|--|------*---------|--|------*------
       *                   *                   *      |     |      *         |  |      *         |  |      *         |  |      *         |  |      *      
5------*-------------------*-------------------*------X-----X------*------.--.--|------*---------|--|------*---------|--|------*---------|--|------*------
       *                   *                   *                   *      |     |      *         |  |      *         |  |      *         |  |      *      
6------*-------------------*-------------------*-------------------*------X-----X------*------.--.--|------*---------|--|------*---------|--|------*------
       *                   *                   *                   *                   *      |     |      *         |  |      *         |  |      *      
7------*-------------------*-------------------*-------------------*-------------------*------X-----X------*------.--.--|------*---------|--|------*------
       *                   *                   *                   *                   *                   *      |     |      *         |  |      *      
8------*-------------------*-------------------*-------------------*-------------------*-------------------*------X-----X------*------.--.--|------*------
       *                   *                   *                   *                   *                   *                   *      |     |      *      
9------*-------------------*-------------------*-------------------*-------------------*-------------------*-------------------*------X-----X------*------



规定逻辑算符:

Z1Z3 , Z1Z4 , ... , Z1Zn
X2X3 , X2X4 , ... , X2Xn


"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from order_1_plotting import set_publication_style, plot_vs_T, plot_vs_n
import os



def gen_symplectic_vec(positions,name,n):
    """
    生成长度为2n的pauli算符辛表示，先X后Z
    比如 : 
    gen_symplectic_vec(positions=[0],name="Z",n=3) = np.array([0,0,0  , 1,0,0],dtype=np.int16)
    gen_symplectic_vec(positions=[2],name="X",n=3) = np.array([0,0,1  , 0,0,0],dtype=np.int16)
    gen_symplectic_vec(positions=[0,2],name="YX",n=3) = np.array([1,0,1  , 1,0,0],dtype=np.int16)
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
        return gen_symplectic_vec([i],name[0],n) + gen_symplectic_vec([j],name[1],n)


def physical_CNOT(n,i,j,symplectic_vec):
    """
    将物理上的CNOTij作用在某个pauli算符symplectic向量上，返回作用结果
    ^运算的含义是: 1^1 = 0 ; 0^1 = 1 ; 1^0 = 1 ; 0^0 = 0 可以理解为第二位为控制位，若第二位为1则第一位反转，若第二位为0则第一位不变
    CNOTij会将Z从j传到i，将X从i传到j
    """
    symplectic_vec[j] ^= symplectic_vec[i]        # X spread
    symplectic_vec[i+n] ^= symplectic_vec[j+n]    # Z spread
    return symplectic_vec


def add_noise(n,symplectic_vec_list,error_parameter_list,p_single,p_double):
    """
    输入旧的错误串，在每一个单比特位置增加一个单比特错误，在每一个相邻双比特位置增加一个双比特错误。
    数据上，每增加一个错误，在symplectic_vec_list中增加一个辛向量，在error_parameter_list中增加一个实数作为噪声强度，单比特强度为p_single，双比特强度为p_double
    """
    # 单比特错误
    for i in range(n):
        for error_type in ["X","Y","Z"]:
            error_parameter_list.append(p_single)
            symplectic_vec_list.append(gen_symplectic_vec([i],error_type,n))
    # 双比特错误
    for i in range(n-1):
        for error_type in [a+b for a in ["X","Y","Z"] for b in ["X","Y","Z"]]:
            error_parameter_list.append(p_double)
            symplectic_vec_list.append(gen_symplectic_vec([i,i+1],error_type,n))
    return symplectic_vec_list,error_parameter_list


def logical_CNOT(n,i,j,symplectic_vec_list,error_parameter_list,p_single,p_double):
    """
    逻辑CNOT门从第i个logical qubit控制第j个logical qubit
    logical_CNOTij = CNOT01 -> CNOT(i+2)(j+2) -> CNOT0(j+2) -> CNOT(i+2)1
    输入旧的噪声辛表示列表，和噪声参数列表，输出新的经过逻辑CNOT之后的两者
    用法: symplectic_vec_list,error_parameter_list = logical_CNOT(n,i,j,symplectic_vec_list,error_parameter_list,p_single,p_double)
    """
    for ii in range(len(symplectic_vec_list)):
        symplectic_vec_list[ii] = physical_CNOT(n,0,1,symplectic_vec_list[ii])
        symplectic_vec_list[ii] = physical_CNOT(n,i+2,j+2,symplectic_vec_list[ii])
    symplectic_vec_list,error_parameter_list = add_noise(n,symplectic_vec_list,error_parameter_list,p_single,p_double)
    for ii in range(len(symplectic_vec_list)):
        symplectic_vec_list[ii] = physical_CNOT(n,0,j+2,symplectic_vec_list[ii])
        symplectic_vec_list[ii] = physical_CNOT(n,i+2,1,symplectic_vec_list[ii])
    symplectic_vec_list,error_parameter_list = add_noise(n,symplectic_vec_list,error_parameter_list,p_single,p_double)
    return symplectic_vec_list,error_parameter_list


def gen_stabilizer_list(n):
    """
    生成针对n个物理比特的QEDC stabilizer辛表示。
    输入: n 为物理qubit个数。
    运算: 构造两个长度为2n的向量，all_X 表示在每个物理比特上施加 X，all_Z 表示在每个物理比特上施加 Z。
    输出: [all_X, all_Z]
    """
    all_X = np.concatenate([np.ones(n, dtype=np.int16), np.zeros(n, dtype=np.int16)])
    all_Z = np.concatenate([np.zeros(n, dtype=np.int16), np.ones(n, dtype=np.int16)])
    return [all_X , all_Z]


def error_detect(n,error_list,stabilizer_list,error_parameter_list):
    """
    使用给定的稳定子对误差串进行对易性检测，并统计权重。
    输入: n 物理比特数；error_list 为所有误差的辛向量列表；stabilizer_list 为稳定子辛向量列表；error_parameter_list 为与每个误差对应的权重/概率幅。
    运算: 对每个误差分别计算与两个稳定子的辛内积 (X·Z + Z·X) mod 2；若任一结果为1表示被检测出，累加到 fail_weight，否则累加到 success_weight。
    输出: fail_weight（被检测出的总权重）与 success_weight（未被检测出的总权重）。
    """
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


def QEDC_PEC_sample(n,symplectic_vec_list,stabilizer_list,error_parameter_list):
    """
    对当前噪声串执行一次“QEDC + PEC”并计算对应的sample overhead。
    输入: n 物理比特数；symplectic_vec_list 为所有噪声辛向量；stabilizer_list 为检测用稳定子；error_parameter_list 为每个噪声项的权重/概率幅。
    运算: 先用 error_detect 计算被检测出的权重 fail_weight 与漏检权重 success_weight，得到成功率 p_success=1-fail_weight；根据PEC放大量 gamma=1+2*success_weight/p_success 计算一次纠错+PEC的 sample=gamma**2/p_success。
    输出: 单次QEDC+PEC步骤的 sample overhead。
    """
    fail_weight, success_weight = error_detect(n,symplectic_vec_list,stabilizer_list,error_parameter_list)
    p_success = 1 - fail_weight
    gamma = 1 + 2 * success_weight / p_success
    return gamma**2 / p_success


def protocol2_sample_cost(n,p_single,p_double,T):
    """
    计算“protocol 2”（周期性QEDC+PEC）的总 sample overhead。
    输入: n 物理比特数；p_single 单比特噪声强度；p_double 双比特噪声强度；T 表示每进行 T 个逻辑 CNOT 后触发一次 QEDC+PEC。
    运算: 
      1) 初始化噪声列表并在所有物理/相邻双比特位置添加一次噪声；
      2) 依次对 GHZ 制备线路的每个逻辑 CNOT（控制位从0到 n-3）调用 logical_CNOT（其中包含门前后的噪声注入）；
      3) 每累计 T 个逻辑 CNOT 就调用 QEDC_PEC_sample 计算该周期的 sample，随后清空噪声列表重新积累；末尾若不足 T 也做一次检测；
    输出: 完成整个 GHZ 制备（n-2 个逻辑 CNOT）所需的总 sample overhead。
    """
    stabilizer_list = gen_stabilizer_list(n)
    symplectic_vec_list = []
    error_parameter_list = []
    symplectic_vec_list,error_parameter_list = add_noise(n,symplectic_vec_list,error_parameter_list,p_single,p_double)
    control_logical_qubit = 0 # 逻辑GHZ态制备线路的control logical qubit从0开始，一直到n-4
    t = 0
    sample = 1
    while (control_logical_qubit < n-3):
        symplectic_vec_list,error_parameter_list = logical_CNOT(n,control_logical_qubit,control_logical_qubit+1,symplectic_vec_list,error_parameter_list,p_single,p_double)
        t += 1
        control_logical_qubit += 1
        if t == T:
            sample *= QEDC_PEC_sample(n,symplectic_vec_list,stabilizer_list,error_parameter_list)
            t = 0
            symplectic_vec_list = []
            error_parameter_list = []
    if t != 0 :
        sample *= QEDC_PEC_sample(n,symplectic_vec_list,stabilizer_list,error_parameter_list)
    return sample


def pure_PEC_sample_cost(n,p_single,p_double):
    """
    计算全程只做PEC（不做QEDC）的 sample overhead。
    输入: n 物理比特数（应对应 [[n,n-2,2]] 码中的 n-2，例如若QEDC+PEC用n=20，此处取n=18作对比）；p_single 单比特噪声强度；p_double 双比特噪声强度。
    运算: 先估算单层噪声权重 W = n*p_single + (n-1)*p_double，再按 (1+2*W) 的放大量叠加 (n-1) 层，得到 sample = (1+2*W)**(2*n-2)。
    输出: 全程仅PEC方案的总 sample overhead。
    """
    W = 3*n*p_single + 9*(n-1)*p_double
    return (1+2*W)**(2*n-2)





if __name__ == "__main__":
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # save_dir = os.path.join(current_dir, "result")
    p_single = 1e-4
    p_double = p_single**2
    set_publication_style()

    # 绘图 1: 固定 n，扫描 T
    n_fixed = 100
    T_min = 1
    T_max = int(0.5/(3*n_fixed*p_single + 9*(n_fixed-1)*p_double)) // 4
    T_list = np.arange(T_min, max(T_min+1, T_max))
    start_time = time()
    protocol2_sample_T = []
    for T in T_list:
        protocol2_sample_T.append(protocol2_sample_cost(n_fixed, p_single, p_double, int(T)))
    pure_sample_T = [pure_PEC_sample_cost(n_fixed-2, p_single, p_double)] * len(T_list)
    end_time = time()
    print("run time (sweep T) = ", end_time - start_time, " s")

    plot_vs_T(T_list, protocol2_sample_T, pure_sample_T, n_fixed, p_single, p_double, logy=True, show=False)

    # 绘图 2: 固定 T，扫描 n
    T_fixed = 1
    n_list = np.arange(50, 201, 10)
    start_time = time()
    protocol2_sample_n = []
    for n_val in n_list:
        protocol2_sample_n.append(protocol2_sample_cost(int(n_val), p_single, p_double, T_fixed))
    pure_sample_n = [pure_PEC_sample_cost(int(n_val)-2, p_single, p_double) for n_val in n_list]
    end_time = time()
    print("run time (sweep n) = ", end_time - start_time, " s")

    plot_vs_n(n_list, protocol2_sample_n, pure_sample_n, T_fixed, p_single, p_double, logy=True, show=False)

    plt.show()
