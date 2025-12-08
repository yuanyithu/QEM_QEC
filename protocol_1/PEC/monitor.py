import time
from functools import wraps
import psutil
import os
from functools import partial
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt


def format_time(seconds):
    """修改时间输出格式"""
    if seconds < 0.001:
        return f"{seconds*1_000_000:.2f} 微秒"
    elif seconds < 1:
        return f"{seconds*1000:.2f} 毫秒"
    elif seconds < 60:
        return f"{seconds:.3f} 秒"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)} 分 {s:.2f} 秒"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)} 小时 {int(m)} 分 {s:.2f} 秒"

def format_bytes(bytes_size):
    """修改内存输出格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def monitor(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        import threading
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录开始时的RSS
        start_rss = process.memory_info().rss
        start_time = time.time()
        
        # 用于存储峰值RSS和控制轮询
        peak_rss = start_rss
        polling = True
        
        def poll_rss():
            nonlocal peak_rss
            while polling:
                current_rss = process.memory_info().rss
                if current_rss > peak_rss:
                    peak_rss = current_rss
                time.sleep(0.01)  # 每10ms轮询一次
        
        # 启动后台轮询线程
        poll_thread = threading.Thread(target=poll_rss, daemon=True)
        poll_thread.start()
        
        # 执行函数
        result = f(*args, **kwargs)
        
        # 停止轮询
        polling = False
        poll_thread.join()
        
        end_time = time.time()
        # 记录结束时的RSS
        end_rss = process.memory_info().rss
        
        # 计算内存变化
        rss_diff = end_rss - start_rss
        
        print("function name : ", f.__name__)
        print("total time : ", format_time(end_time - start_time))
        # print("start RSS: ", format_bytes(start_rss))
        # print("end RSS: ", format_bytes(end_rss))
        print("peak RSS: ", format_bytes(peak_rss))
        # print("RSS diff: ", format_bytes(rss_diff) if rss_diff >= 0 else f"-{format_bytes(abs(rss_diff))}")
        return result
    return wrapper



if __name__ == "__main__":


    I2 = jnp.array([[1+0j,0],[0,1]])
    X = jnp.array([[0+0j,1],[1,0]])
    Y = jnp.array([[0,-1j],[1j,0]])
    Z = jnp.array([[1+0j,0],[0,-1]])

    # @monitor
    def ordered_gen(n):
        result = 0.5*I2+0.5*Y
        for i in range(2,n+1):
            result = jnp.einsum("ij,kl->ikjl",result,0.5*I2+0.5*Y).reshape([2**i,2**i])
        return result

    # @monitor
    def clever_gen(n):
        # print("n = ",n)
        if n == 1:
            return 0.5*I2+0.5*Y
        elif n%2 == 0:
            return jnp.einsum("ij,kl->ikjl",clever_gen(n//2),clever_gen(n//2)).reshape([2**n,2**n])
        elif n%2 == 1:
            return jnp.einsum("ij,kl,mn->ikmjln",clever_gen(n//2),clever_gen(n//2),clever_gen(1)).reshape([2**n,2**n])

    @partial(jax.jit, static_argnums=(0,))
    def fast_gen(n):
        return clever_gen(n)

    n = 16
    start = time.time()

    # A = ordered_gen(n)
    # print("A.sum() = ",A.sum())
    # end = time.time()
    # print(format_time(end-start))

    B = fast_gen(n)
    print("B.sum() = ",B.sum())
    print("done")
    end = time.time()
    print(format_time(end-start))