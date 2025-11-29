from PEC_base import *
from visualization import *
from functools import partial

@partial(jax.jit, static_argnames=('n'))
def total_PEC(n,p,std,L,key):
    error_matrix = gen_error(p,std,n,key)
    layered_error_matrix = gen_layered_error(error_matrix,L)
    NL = tensor_single_pauli_noise(n,layered_error_matrix)
    NL_matrix = jnp.einsum("ijkl->ikjl",NL).reshape([4**n,4**n])
    NL_inv_matrix = jnp.linalg.pinv(NL_matrix,rtol=1e-10)
    NL_inv = jnp.einsum("ikjl->ijkl",NL_inv_matrix.reshape([2**n,2**n,2**n,2**n]))
    return PEC_sample_cost(n,NL_inv)

@partial(jax.jit, static_argnames=('n'))
def layered_PEC(n,p,std,L,key):
    error_matrix = gen_error(p,std,n,key)
    N = tensor_single_pauli_noise(n,error_matrix)
    N_matrix = jnp.einsum("ijkl->ikjl",N).reshape([4**n,4**n])
    N_inv_matrix = jnp.linalg.pinv(N_matrix,rtol=1e-10)
    N_inv = jnp.einsum("ikjl->ijkl",N_inv_matrix.reshape([2**n,2**n,2**n,2**n]))
    result = PEC_sample_cost(n,N_inv)**L
    return result

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    n = 4
    p = 0.0003
    std = 0
    
    total_sample = []
    layered_sample = []
    L_list = jnp.arange(1,3003,100)
    for L in L_list:
        total_sample.append(total_PEC(n,p,std,L,key))
        layered_sample.append(layered_PEC(n,p,std,L,key))
        # print("---------------")
        # print("L = ",L)
        # print("total sample = ",jnp.log10(total_sample[-1]))
        # print("layered sample = ",jnp.log10(layered_sample[-1]))
        # print("---------------")
    
    combined = jnp.column_stack([L_list, np.log10(total_sample), np.log10(layered_sample)])
    np.savetxt('output.csv', combined, delimiter=',', fmt='%s')


    # 设置画布大小，使其更清晰
    plt.figure(figsize=(10, 6), dpi=100)
    # 绘制曲线，增加 marker 以便看清具体采样点，linewidth 增加线条粗细
    plt.plot(L_list, total_sample, label="Total PEC", marker='o', markersize=6, linewidth=2)
    plt.plot(L_list, layered_sample, label="Layered PEC", marker='s', markersize=6, linewidth=2)
    # --- 核心修改：设置对数轴并优化刻度显示 ---
    ax = plt.gca() # 获取当前坐标轴对象
    ax.set_yscale('log') # 设置纵轴为对数坐标
    # 设置纵轴刻度格式：
    # 使用 ScalarFormatter 强制显示为普通数字而不是 10^x 形式
    # formatter = ticker.ScalarFormatter() 
    # formatter.set_scientific(False) # 关闭科学计数法
    # ax.yaxis.set_major_formatter(formatter)

    # 如果你也希望显示次级刻度（比如 0.2, 0.3...），取消下面这行的注释
    ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

    # --- 3. 装饰与标注 ---
    
    # 添加网格：which='both' 表示同时显示主刻度和次刻度的网格
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    # 添加轴标签和标题 (支持 LaTeX 格式)
    plt.xlabel(r"Number of Layers ($L$)", fontsize=12)
    plt.ylabel("PEC Value (Log Scale)", fontsize=12)
    plt.title(f"PEC Comparison: Total vs Layered (n={n}, p={p})", fontsize=14)

    # 优化图例
    plt.legend(frameon=True, fontsize=11, loc='best')

    # 防止标签被截断
    plt.tight_layout()
    save_kwargs = {'bbox_inches': 'tight'}
    plt.savefig('PEC_Comparison_Final_300dpi.png', format='png', dpi=300, **save_kwargs)
    plt.show()