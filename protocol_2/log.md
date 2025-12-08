## 1 项目总览

总体目的：结合error detection和PEC方案，设计新型抵抗噪声protocol.

核心物理intuition：注意到频繁进行error detection并抛弃掉错误的分支，带来的采样复杂度上升，比不断做PEC带来的采样复杂度上升要快。

项目任务：设计出**可扩展**的protocol，写出代码进行模拟，希望观察到sample cost符合测量越频繁越小的预期结果。

## 2 日志

------

### a.0 原始版本

- protocol：进行T次noisy gate之后直接error detection，然后对剩下部分开展PEC.
- 噪声模型：independent local depolarizing.
- 实验结果：在 (p=0.0003 , n=6 , L=3000 , T=1/10/50/100/500) 的情况进行了数值模拟，观察到在T<1000的范围内明显T越小，总的sample cost越小
- 不足：需要计算的各部分复杂度极高，数值模拟只能开展到n=6，方案不可扩展
- 思考：没有利用噪声的稀疏性质，应结合IBM 2022 sparse PEC的噪声模型与protocol，设计可扩展protocol

------

### a.1 修改噪声模型为sparse local

- protocol：与原始版本一样，经过T层noisy gate之后error detect，post select之后对剩下部分开展PEC.
- 噪声模型：每层噪声改为IBM 2022文献中的sparse local noise乘积的形式
- 实验结果：经过理论分析(`A13-5.pdf`)，error propagation、error detection success probability、reduced noise channel、PEC channel等部分计算复杂度仍然很高，最后几部分复杂度与原始版本无明显差异。因此并未写代码开展数值模拟。
- 不足：经典计算复杂度高
- 思考：没有像2022 IBM文献一样很好利用sparse noise的乘积形式进行PEC. 当前为了处理error detection之后的存活概率、reduced channel、PEC channel等，还必须将乘积形式的噪声完全拆开乘出来，造成指数多项，而从reduced channel计算PEC channel这部分也因为丧失了乘积的形式，也变得经典上低效。有什么办法能保持乘积形式？

------

### a.2 头脑风暴

- 想法1：Monte Carlo采样
  - protocol：保留各层噪声乘积形式，进行逆的过程本来也是概率性采样，对相乘的每一项采样之后给出一个整体作用的算符，再去与stabilizer比对判断是否通过了error detection
  - 数学表达式：
  - 不确定与潜在不足：能否保证无偏？能否保证是逆信道？如何计算sample cost？
- 想法2：张量网络
  - 借鉴：yuchen guo & shuo yang 2022 基于张量网络的QEM文章，利用各层噪声的local与sparse，处理多层结果
  - 困难：需要学习论文
- 下一步计划：先采用Monte Carlo采样思路，看看能不能设计出合理protocol，且可扩展。

------

### a.3 稀疏噪声按错误率展开近似就错

- protocol: 噪声理解为概率性在n个qubit和T层的每个可能位点按概率发生单比特、双比特噪声。还是一样propagate到T层结束进行error detection，将错误率视为小量，容易给出近似到1阶的reduced noise channel，也容易通过采样执行精确到错误率1阶的PEC操作。也容易扩展到2阶reduced noise channel，和2阶PEC.
- 目前先关注1阶，将模拟程序完成。
- 不足：在关心的n=20,p=0.0003,T=100区间，发生错误的项数*p约为10，此时不再可以将所有噪声视为一阶微扰处理。如果之后希望算法可扩展，且噪声关注0.0003区间，则一阶修正必须要求T足够小才可。换句话说，微扰论成立的区间是9nTp << 1的区间，可扩展按n为$10^2$计算，p按照$10^{-4}$计算，则T必须在个位数，才勉强保证微扰论成立，除非考虑噪声再小一个数量级的区间，不过也只能将我们的算法限定在这个近似区间了，否则任何阶的微扰论都不再成立。
- debug：发现了程序的一系列逻辑错误，修正了这些错误。
- 思考1 - 噪声模型设定：用当前噪声模型与之前的iid local depolarizing noise对比是不公平的。因为初版程序中单比特、双比特噪声参数一样，这一方面与之前噪声模型不同，不具备可比性；另一方面不一定符合现实。需要结合IBM paper中的部分数值，或者至少按照双比特错误率约为单比特错误率的平方的方式，与iid local depolarizing noise去对比，便于验证程序正确性。
- 思考2 - 逻辑门：之前采用的线路不正确，如果只是按照逐层作用CNOT，并不保持stabilizer group，之后需要做门的时候必须利用逻辑门编译之后的物理门进行作用。

------

### a.4 使用1阶PEC，并计算

- protocol：稀疏噪声，1阶PEC修正，每间隔T层执行一次。时刻监控近似合理性。不做门
- 实验结果：观察到符合预期的曲线，总的sample cost随着T降低而降低，并且呈现sample ~ $e^{(a+bT)}$的极其严格的拟合关系（没做门是一个原因）。并且经典计算高效，是多项式的，PC上分钟级别时间能算到n=1000
- 不足1：目前没做门，很容易推广到clifford门（注意需保持stabilizer group，为逻辑门）
- 不足2：non-clifford gate如何处理
- 不足3：未考虑测量错误，syndrome引入的错误
- 思考1：应该选择一个什么样的纯clifford的量子线路作为展示？有用的线路（例如VQE）都是non-clifford的，不如生成随机CNOT线路，但注意保证子空间结构，一个可能的做法是执行逻辑CNOT。
- 思考2：针对不足2，一种可能的做法是每当出现magic gate就执行QEDC+PEC. 需要注意，由non-clifford gate引入的coherent error是无法被PEC完全纠正的
- 思考3：测量错误是大问题，一定会被问。据说传统PEC有办法解决测量错误，需调研。
- 下一步计划：先解决不足1的clifford门这一部分，选择随机CNOT逻辑线路（或能否理论计算某种随机线路平均效果？），编译成为多层物理线路，保证全是clifford gate，在此基础上运行我们的算法

------

### b.0 与Yuchen Guo讨论

- 关于噪声：更物理的假设是，首先一定是门带来噪声，其次不妨假设做单比特门时没有噪声，做双比特门时存在单比特噪声为主、双比特噪声为辅。
- 浅层线路用张量网络表示是高效的，计算后选择概率、求逆信道（需做一个变分法）随着n也是高效的
- 一个一定会被问的问题：测量引入的噪声如何考虑？可以考虑利用一些local的check
- 存在non-clifford门怎么办？可以假设都是T门，而T门对pauli影响很小（后续需仔细考虑）
- 实际上PEC解决不了coherent error成分（这是PEC的一大固有不足），这里引入error detection有可能能提高克服coherent error的效果，可作为一个亮点。
- 我们文章在自己定义问题，完全可以找出自己相比PEC好的地方即可

------

### b.1 使用张量网络描述浅层线路，不用近似

- protocol：用张量网络描述浅层线路，benchmark过程直接利用Guo文章提到的MPO tomography argue说可以高效完成，实际的工作只需要直接生成一个tensor network表示的noisy量子线路，然后缩并计算存活概率，进而计算剩余信道，接着变分法求逆信道，然后截断取PEC操作，计算PEC采样复杂度。
- coherent error仍然是一个问题，magic操作带来的coherent error如果能够利用这种方法可以被更好处理（相比于传统PEC），将是一个很好的点。

