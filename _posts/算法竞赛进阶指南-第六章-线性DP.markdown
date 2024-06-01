---
title: 算法竞赛进阶指南-第六章-线性DP
date: 2023-03-09 22:07
layout: post
category: ACM
tags:
    - markdown
    - components
    - extra
author: Zhenshuai Yin
description: 算法竞赛进阶指南-第六章-线性DP
hidden: false
image: /assets/images/markdown.jpg
headerImage: false
---

# 动态规划

动态规划算法把原问题视为若干个重叠字问题的逐层递进,每个子问题的求解过程都构成一个"阶段".在完成前一个阶段的计算后,动态规划才会执行下一阶段的计算.

为保证这些计算能够按顺序,不重复的进行,动态规划要求已经求解的子问题不受后续阶段的影响,这个条件也被叫做"无后效性".即动态规划对状态空间的遍历构成一张有向无环图,遍历顺序就是该有向无环图的一个拓扑序.有向无环图中的节点对应问题中的"状态",图中的边对应状态之间的"转移",转移的选取就是动态规划中的"决策"

动态规划用于求解最优化问题时,下一阶段的最优解应该能够由前面各阶段子问题的最优解导出.这个条件被称为"最优子结构性质".

"状态""阶段"和"决策"是构成动态规划算法的三要素,而"子问题重叠性","无后效性"和"最优子结构性质"是问题能用动态规划求解的三个基本条件

动态规划算法把相同的计算过程作用于各阶段的同类子问题,因此,我们只需要定义出DP的计算过程,就可以编程实现.这个计算过程就被称为"状态转移方程"

## 线性DP

具有线性"阶段"划分的动态规划算法被统称为线性DP

![image-20230310110612621](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230310110614.png)

![image-20230310110855601](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230310110856.png)

![image-20230310111104021](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230310111104.png)

在这类问题中,需要计算的对象表现出明显的维度以及有序性,每个状态的求解直接构成一个阶段,这使得DP的状态表示就是阶段的表示.因此,我们只需要在每个维度上各取一个坐标值作为DP的状态,自然就可以描绘出"已求解部分"在状态空间中的轮廓特征,该轮廓的进展就是阶段的推移.

###### Mr.Young's Picture Peremutations

线性DP,先考虑好状态转移方程,然后编程实现

![image-20230310115801080](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230310115802.png)

![image-20230310115809583](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230310115810.png)

```c++
#include <bits/stdc++.h>
using namespace std;
int k, num[5];
long long f[31][31][31][31][31];
int main()
{
    cin >> k;
    while (k)
    {
        memset(f, 0, sizeof(f));
        f[0][0][0][0][0] = 1;
        memset(num, 0, sizeof(num));
        for (int i = 0; i < k; i++)
            cin >> num[i];
        for (int a = 0; a <= num[0]; a++)
            for (int b = 0; b <= min(a, num[1]); b++)
                for (int c = 0; c <= min(b, num[2]); c++)
                    for (int d = 0; d <= min(c, num[3]); d++)
                        for (int e = 0; e <= min(d, num[4]); e++)
                        {
                            long long &x = f[a][b][c][d][e];
                            if (a && a - 1 >= b)
                                x += f[a - 1][b][c][d][e];
                            if (b && b - 1 >= c)
                                x += f[a][b - 1][c][d][e];
                            if (c && c - 1 >= d)
                                x += f[a][b][c - 1][d][e];
                            if (d && d - 1 >= e)
                                x += f[a][b][c][d - 1][e];
                            if (e)
                                x += f[a][b][c][d][e - 1];
                        }
        cout << f[num[0]][num[1]][num[2]][num[3]][num[4]] << endl;
        cin >> k;
    }
}
```

