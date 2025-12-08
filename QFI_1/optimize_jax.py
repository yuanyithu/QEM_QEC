from single_round_jax import *
import matplotlib.pyplot as plt

def proj(Q, A):
    """投影A到Q的切空间（Stiefel流形）"""
    return A - 0.5 * Q @ (A.T.conj() @ Q + Q.T.conj() @ A)

def norm(A):
    """计算Frobenius范数"""
    return jnp.sqrt(jnp.real(jnp.trace(A.T.conj() @ A)))

def transport(Q, V):
    """速度向量的平行传输"""
    V_proj = proj(Q, V)
    norm_V = norm(V)
    norm_V_proj = norm(V_proj)
    # 避免除零
    if norm_V_proj < 1e-10:
        return jnp.zeros_like(V)
    return V_proj * (norm_V / norm_V_proj)

def retraction(Q):
    """使用QR分解将Q投影回Stiefel流形"""
    Q_new, R = jnp.linalg.qr(Q)
    # 保持相位一致性
    signs = jnp.sign(jnp.diag(R))
    signs = jnp.where(signs == 0, 1, signs)
    return Q_new * signs

def optimize_riemannian(
    Q_init, 
    p, 
    n_steps=200, 
    lr=0.01, 
    gamma=0.001, 
    decay=0.9,
    verbose=True,
    key_seed=42
):
    """
    黎曼共轭梯度优化
    
    参数:
        Q_init: 初始正交矩阵 (复数)
        p: 噪声参数
        n_steps: 迭代步数
        lr: 学习率 (梯度步长)
        gamma: 随机力强度
        decay: 速度衰减率
        verbose: 是否打印信息
        key_seed: 随机种子
    
    返回:
        Q_final: 优化后的矩阵
        costs: 每步的cost值
    """
    Q = Q_init
    N, K = Q.shape
    V = jnp.zeros_like(Q)
    
    # 计算n和k
    n = int(jnp.log2(N))
    k = int(jnp.log2(K))
    
    # 定义梯度函数（对Q_real和Q_imag求梯度）
    grad_fn = jit(grad(max_eig, argnums=(0, 1)),static_argnums=(2, 3))
    
    key = PRNGKey(key_seed)
    costs = []
    
    for step in range(n_steps):
        Q_real = jnp.real(Q)
        Q_imag = jnp.imag(Q)
        
        # 计算当前cost
        cost = max_eig(Q_real, Q_imag,n,k, p)
        costs.append(float(cost))
        
        # 计算梯度（梯度上升方向）
        grad_real, grad_imag = grad_fn(Q_real, Q_imag,n,k, p)
        nabla = grad_real + 1j * grad_imag


        # 投影梯度到切空间
        nabla = proj(Q, nabla)
        
        # 生成随机扰动
        key, subkey = random.split(key)
        key_real, key_imag = random.split(subkey)
        delta_real = random.normal(key_real, shape=(N, K))
        delta_imag = random.normal(key_imag, shape=(N, K))
        delta = delta_real + 1j * delta_imag
        
        # 投影并归一化随机扰动
        delta = proj(Q, delta)
        delta_norm = norm(delta)
        if delta_norm > 1e-10:
            delta = delta / delta_norm
        
        # 传输旧速度到当前切空间
        V_old = transport(Q, V)
        
        # print("-------------------------")
        # print("step ",step)
        # print("norm(grad_real) = ",norm(grad_real))
        # print("norm(grad_imag) = ",norm(grad_imag))
        # print("norm(delta) = ",norm(delta))
        # print("norm(nabla) = ",norm(nabla))
        # print("norm(V_old) = ",norm(V_old))
        # print("-------------------------")
        # print()

        # 更新速度（梯度上升 + 随机力 + 动量）
        V = gamma * delta + lr * nabla + decay * V_old
        
        # 在切空间移动
        Q = Q + V
        
        # Retraction回Stiefel流形
        Q = retraction(Q)
        
        # 打印进度
        if verbose and (step % 20 == 0 or step == n_steps - 1):
            print(f"Step {step:4d}, Cost: {cost:.8f}, |V|: {norm(V):.6f}")
    
    return Q, jnp.array(costs)


def plot_optimization(costs, save_path=None):
    """可视化优化过程"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (max eigenvalue)', fontsize=12)
    plt.title('Riemannian Conjugate Gradient Optimization', fontsize=14)
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def show_state(Q, n):
    """
    打印 Q 中定义的逻辑基态的详细信息。

    参数:
        Q (jnp.ndarray): 形状为 (N, M) 的复数矩阵 (N=2**n, M=2**k)。
        n (int): 物理qubit的数量。
    """
    N, M = Q.shape
    # 遍历 M (M=2**k) 个逻辑基态
    for j in range(M):
        state_str = ""
        first_term = True
        # 遍历 N (N=2**n) 个计算基
        for i in range(N):
            c = Q[i, j]
            # 设置一个阈值，只显示系数绝对值大于 1e-6 的项
            if jnp.abs(c) > 1e-6:
                real_part = jnp.real(c)
                imag_part = jnp.imag(c)
                # 格式化系数: (real.xxx sign imag.xxx i)
                # :.3f 保留3位小数
                # :+.3f 强制显示正负号，并保留3位小数
                coeff_str = f"({real_part:.3f}{imag_part:+.3f}i)"
                # 格式化基态: |00...0>
                # :0{n}b 表示格式化为n位的二进制数，不足n位则在前面补0
                basis_str = f"|{i:0{n}b}>"
                if first_term:
                    # 第一项不加 " + "
                    state_str = f"{coeff_str} {basis_str}"
                    first_term = False
                else:
                    # 后续项前面加上 " + "
                    state_str += f" + {coeff_str} {basis_str}"
        if first_term:
            # 如果 first_term 仍然为 True，说明这是一个零向量
            print(f"|e{j+1}> = 0.0")
        else:
            print(f"|e{j+1}> = {state_str}")






if __name__ == "__main__":
    # 参数设置
    p = 0
    k = 1
    n = 2
    
    # 初始化Q（使用trivial_Q或随机正交矩阵）
    print("=" * 50)
    print("使用trivial初始化")
    Q_init = trivial_Q(n)
    Q_real = jnp.real(Q_init)
    Q_imag = jnp.imag(Q_init)
    initial_cost = max_eig(Q_real, Q_imag,n,k, p)
    print(f"Initial cost: {initial_cost:.8f}")
    print("=" * 50)
    
    # 运行优化
    Q_opt, costs = optimize_riemannian(
        Q_init=Q_init,
        p=p,
        n_steps=3,
        lr=0.02,
        gamma=0.005,
        decay=0.9,
        verbose=True
    )
    
    # 验证最终结果
    Q_real_opt = jnp.real(Q_opt)
    Q_imag_opt = jnp.imag(Q_opt)
    final_cost = max_eig(Q_real_opt, Q_imag_opt,n,k, p)
    
    print("=" * 50)
    print(f"Final cost: {final_cost:.8f}")
    print(f"Improvement: {final_cost - initial_cost:.8f}")
    
    # 验证正交性
    orthogonality_error = jnp.max(jnp.abs(Q_opt.T.conj() @ Q_opt - jnp.eye(2**k)))
    print(f"Orthogonality error: {orthogonality_error:.2e}")
    print("=" * 50)
    # show_state(Q_opt,n)


    # 绘制优化曲线
    # plot_optimization(costs)