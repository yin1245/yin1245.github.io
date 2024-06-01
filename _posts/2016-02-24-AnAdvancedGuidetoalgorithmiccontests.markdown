---
title: "An Advanced Guide to algorithmic contests - Chapter 3"
layout: post
date: 2016-02-24 22:48
image: /assets/images/markdown.jpg
headerImage: false
tag:
- markdown
- components
- extra
category: FedMD
hidden: false
author: yin1245
description: An Advanced Guide to algorithmic contests - Chapter 3
---

# 搜索

主要是讲述了深度优先搜索和广度优先搜索的各种用法,以及其各种变形.还有启发式搜索(好像也是基于深搜和广搜)

## 树与图的遍历

### 树与图的深度优先遍历,树的DFS序,深度和重心

![image-20230127173244557](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529.png)

#### 树的DFS序

![image-20230127173341232](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-1.png)

#### 树的重心

![image-20230127173839710](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-2.png)

#### 图的连通块划分

![image-20230127174036553](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-3.png)

### 树与图的广度优先遍历,拓扑排序

#### 广度优先遍历

![image-20230127174453136](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-4.png)

#### 拓扑排序

![image-20230127174607871](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-5.png)

拓扑排序可以判定有向图中是否存在环.若拓扑序列长度小于图中点的数量,则说明图中存在环

<img src="https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-6.png" alt="image-20230127175359507" style="zoom:50%;" />

### 例题

#### 可达性统计

从x出发能够到达的点是从x的各个后继节点y出发能够到达的点的并集,再加上x自身.

先用拓扑排序算法求出一个拓扑序,然后按照拓扑序的倒序进行计算

利用状态压缩,可以使用一个N为二进制数存储每个f(x),最后,每个f(x)中1的个数就是从x出发能够到达的节点数量

时间复杂度为N(N+M)/32

```c++
#include <bits/stdc++.h>
using namespace std;
vector<int> lu[30005];
int deg[30005];
int n, m;
int a[30005];
int cnt = 0;
void topsort()
{
    queue<int> q;
    for (int i = 1; i <= n; i++)
    {
        if (deg[i] == 0)
            q.push(i);
    }
    while (q.size())
    {
        int x = q.front();
        q.pop();
        a[++cnt] = x;
        for (int i = 0; i < lu[x].size(); i++)
        {
            int y = lu[x][i];
            if (--deg[y] == 0)
                q.push(y);
        }
    }
}
bitset<30005> s[30005];

int main()
{

    cin >> n >> m;
    for (int i = 0; i < m; i++)
    {
        int x, y;
        cin >> x >> y;
        lu[x].push_back(y);
        deg[y]++;
    }
    topsort();
    for (int i = n; i >= 1; i--)
    {
        s[a[i]][a[i]] = 1; // 将本身所在的点标注为1
        for (int j = 0; j < lu[a[i]].size(); j++)
        {
            s[a[i]] |= s[lu[a[i]][j]];//将其后继节点合并
        }
    }
    for (int i = 1; i <= n; i++)
    {
        cout << s[i].count() << '\n';//setbit的函数,输出1的数量
    }
    return 0;
}
```

## 深度优先搜索

子集和,全排列,N皇后问题均是可以用深搜求解的经典NPC问题

#### 小猫爬山

尝试依次把每一只小猫分配到一辆已经租用的缆车上,或者新租一辆缆车安置这只小猫.

状态:已经运送的小猫有多少只,已经租用的缆车有多少辆,每辆缆车上当前搭载的小猫重量之和

如果在搜索的任何时刻发现cnt已经大于或等于已经搜到的答案,那么当前分支就可以立即回溯

重量较大的小猫明显比重量较轻的小猫更难运送,还可以在整个搜索前把小猫按重量递减排序,优先搜索重量较大的小猫,减少搜索树分支的数量

```c++
#include <bits/stdc++.h>
using namespace std;
int n;
int c[20];
int w;
int che[20];
int cnt = 1;
int ans = 30;
void dfs(int now)
{
    if(cnt>ans)
        return;
    if(now > n) 
    {
        ans = min(ans, cnt);
        return;
    }
    for (int i = 1; i <= cnt;i++)
    {
        if(che[i]+c[now] <= w)
        {
            che[i] += c[now];
            dfs(now + 1);
            che[i] -= c[now];
        }
    }
    cnt++;
    che[cnt] += c[now];
    dfs(now + 1);
    che[cnt] -= c[now];
    cnt--;
}
int main()
{
    cin >> n >> w;
    for (int i = 1; i <= n;i++)
        cin >> c[i];
    sort(c+1, c + n+1, greater<int>());
    dfs(1);
    cout << ans;
    return 0;
}
```

#### Sudoku

状态为数独的每个位置上填了什么数

分支为在还没有填的位置上,可填写的合法的数字

边界为:所有位置都被填满;存在某个位置没有能填的合法数字

在任意一个状态下,只需要找出一个位置,考虑该位置上填什么数,不需要枚举所有的位置和可填的数字

在搜索算法中,采取:在每个状态下,从所有未填的位置里选择"能填的合法数字"最少的位置的策略,而不是任意找一个位置

对于每行每列每个九宫格,分别用一个9位二进制数保存那些数字还可以填

对于每个位置,把它所在行,列,九宫格的3个二进制数做位与运算,就可以得到该位置能填哪些数(用lowbit运算可以把能填的数字取出)

回溯

```c++
#include<bits/stdc++.h>
using namespace std;
struct SHU
{
    int num = 0;
    int hang;
    int lie;
    int jiu;
} shu[90];
int hang[10], lie[10], jiu[10],cnt[515];
int H[1 << 10 + 1];
bool jud;
void dfs()
{
    if(jud)
        return;
    int minn = 0;
    int mincnt = 10;
    int cntt = 0;
    for (int i = 1; i <= 81;i++)
    {
        if(shu[i].num == 0)
        {
            cntt++;
            int temp = hang[shu[i].hang] & lie[shu[i].lie] & jiu[shu[i].jiu];
            int t = cnt[temp>>1];
            if(t == 0)
                return;
            if(t < mincnt)
            {
                mincnt = t;
                minn = i;
            }
        }
    }
    // cout << minn<<' ';
    if(!cntt)
    {
        for (int i = 1; i <= 81;i++)
            cout << shu[i].num;
        cout << '\n';
        jud = 1;
        return;
    }
    else
    {
        if(!mincnt)
            return ;
        else
        {
            int temp = hang[shu[minn].hang] & lie[shu[minn].lie] & jiu[shu[minn].jiu];
            while(temp > 0)
            {
                int t = temp & -temp;
                temp -= t;
                hang[shu[minn].hang] -= t;
                lie[shu[minn].lie] -= t;
                jiu[shu[minn].jiu] -= t;
                shu[minn].num = H[t];
                // for (int i = 1; i <= 81;i++)
                //     cout << shu[i].num;
                // cout << '\n';
                dfs();
                hang[shu[minn].hang] += t;
                lie[shu[minn].lie] += t;
                jiu[shu[minn].jiu] += t;
                shu[minn].num = 0;
            }
        }
    }
    return ;
}

int main()
{
    for (int i=0; i<1<<9; i++)
        for (int j=i; j; j-= (j & -j))
            cnt[i]++;
    for (int i = 0; i <= 10;i++)
        H[1 << i] = i;
    int now = 1;
    for (int i = 1; i <= 9;i++)
        for (int j = 1; j <= 9;j++)
        {
            int choice;
            if(i<=3)
                choice = 0;
            else if(i<=6)
                choice = 3;
            else
                choice = 6;

            if(j<=3)
                choice += 1;
            else if(j<=6)
                choice += 2;
            else
                choice += 3;
            shu[now].jiu = choice;
            shu[now].hang = i;
            shu[now++].lie = j;
        }
    string str;
    cin >> str;
    while(str != "end")
    {
        jud = 0;
        for (int i = 0; i <= 81;i++)
            shu[i].num = 0;
        for (int i = 1; i <= 9; i++)
        {
            hang[i] = lie[i] = jiu[i] = 0b1111111110;
        }
        for (int i = 0; i < str.length(); i++)
        {
            if (str[i] != '.')
            {
                shu[i + 1].num = str[i] - '0';
                hang[shu[i + 1].hang] -= 1 << (str[i] - '0');
                lie[shu[i + 1].lie] -= 1 << (str[i] - '0');
                jiu[shu[i + 1].jiu] -= 1 << (str[i] - '0');
            }
        }
        dfs();
        cin >> str;
    }
    return 0;
}
```

## 剪枝

搜索算法面对的状态可以看做一个多元组,其中每一元都是问题状态空间中的一个"维度".

搜索过程中的剪枝,其实就是针对每个"维度"与该维度的边界条件,加以缩放,推导,得出一个相应的不等式,来减少搜索树分支的扩张

为了进一步提高剪枝的效果,除了当前花费的"代价"之外,我们还可以对未来至少需要花费的代价进行预算,这样更容易接近每个维度的上下界

在一般的剪枝不足以应对问题的时候,也可以结合各维度之间的联系得到更加精准的剪枝.

优化搜索顺序

排除等效冗余

可行性剪枝

最优性剪枝

记忆化

#### Sticks

从小到大枚举原始木棒的长度len(需要是所有木棍长度总和sum的约数)

状态:已经拼好的原始木棍数,正在拼的原始木棍的长度,每个木棍的使用情况

每个状态下,从尚未使用的木棍中选择一个,尝试拼到当前的原始木棍里,然后递归新的状态,边界为拼好sum/len根原始木棍,或者因无法继续拼接而失败

剪枝:

木棍长度从大到小排序

限制先后加入一根原始木棒的木棍长度是递减的

......

```c++
#include <bits/stdc++.h>
using namespace std;
int a[100], v[100], n, len, cnt;
bool dfs(int stick, int cab, int last)
{
    if (stick > cnt)
        return true;
    if (cab == len)
        return dfs(stick + 1, 0, 1);
    int fail = 0;
    for (int i = last; i <= n; i++)
    {
        if (!v[i] && cab + a[i] <= len && fail != a[i])
        {
            v[i] = 1;
            if (dfs(stick, cab + a[i], i + 1))
                return true;
            fail = a[i];
            v[i] = 0;
            if (cab == 0 || cab + a[i] == len)
                return false;
        }
    }
    return false;
}
int main()
{
    while(cin >> n && n)
    {
        int sum = 0, val = 0;
        for (int i = 1; i <= n;i++)
        {
            cin >> a[i];
            sum += a[i];
            val = max(val, a[i]);
        }
        sort(a + 1, a + n + 1, greater<int>());
        for(len = val; len <= sum;len++)
        {
            if(sum % len)
                continue;
            cnt = sum / len;
            memset(v, 0, sizeof(v));
            if(dfs(1,0,1))
                break;
        }
        cout << len << endl;
    }
    return 0;
}
```

#### 生日蛋糕

搜索框架:从下往上搜索,枚举每层的半径和高度作为分支

搜索状态:第几层,当前外表面面积,当前体积,当前层下面一层的高度和半径

剪枝:

上下界剪枝:限定枚举半径和高度

优化搜索顺序,使用倒序枚举

可行性剪枝:预处理出从上往下前i层的最小体积和侧面积,如果当前体积加上面层的最小体积大于N,可以剪枝

最优性剪枝:

如果当前表面积加上上面层的最小侧面积大于已经搜到的答案,剪枝

![image-20230201100903577](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-7.png)

```c++
#include <bits/stdc++.h>
using namespace std;
int n, m;
int h[25], r[25], min_v[25], min_c[25];
int ans = 1e9;
int v = 0, s = 0;
void find(int dep)
{

    for (r[dep] = min((int)floor(sqrt(n - v)), r[dep + 1] - 1); r[dep] >= dep; r[dep]--)
        for (h[dep] = min((int)floor((n - v) / (r[dep] * r[dep])), h[dep + 1] - 1); h[dep] >= dep; h[dep]--)
        {
            if (dep == m)
            {
                v = s = 0;
                s += r[dep] * r[dep];
            }
            v += r[dep] * r[dep] * h[dep];
            s += 2 * r[dep] * h[dep];
            if (v + min_v[dep - 1] > n || s + min_c[dep - 1] > ans || 2 * (n - v) / r[dep] + s > ans)
            {
                v -= r[dep] * r[dep] * h[dep];
                s -= 2 * r[dep] * h[dep];
                continue;
            }
            if (dep == 1 && v == n)
            {
                ans = min(ans, s);
            }
            if (dep > 1)
                find(dep - 1);
            v -= r[dep] * r[dep] * h[dep];
            s -= 2 * r[dep] * h[dep];
        }
}
int main()
{
    cin >> n >> m;
    min_v[1] = 1, min_c[1] = 2;
    for (int i = 2; i <= m; i++)
    {
        min_v[i] = min_v[i - 1] + i * i * i;
        min_c[i] = min_c[i - 1] + 2 * i * i;
    }
    h[m + 1] = r[m + 1] = 1e9;
    find(m);
    if (ans != 1e9)
        cout << ans;
    else
        cout << 0;
    return 0;
}
```

#### Sudoku

略(没做)

## 迭代加深

从小到大限制搜索的深度,如果在当前深度限制下搜不到答案,就把深度限制增加,重新进行一次搜索,这就是迭代加深思想.

当搜索树规模随着层次的深入增长很快,并且我们能够确保答案在一个较浅层的节点时,就可以采用迭代加深的深度优先搜索算法来解决问题.

#### Addition Chains

搜索框架:依次搜索序列中的每个位置k,枚举i和j作为分支,把x[i]和x[j]的和填到x[k]上,然后递归填写下一个位置

剪枝:

优化搜索顺序:为了让序列中的数尽快逼近n,在枚举i和j时,从大到小枚举

排除等效冗余:对于不同的i和j,和可能是相等的,可以用一个bool数组进行判重

序列长度不会太大,而每次枚举两个数的和,分支很多,所以采用迭代加深的搜索方式,从1开始限制搜索深度,若搜索失败,就增加深度限制重新搜索,直到找到一组解时,输出答案

```c++
#include <bits/stdc++.h>
using namespace std;
int n;
int a[15];
int len;
bool b[105];
bool ans = 0;
void find(int now, int g)
{
    if (ans)
        return;
    if (now > g)
        return;
    for (int i = now - 1; i >= 1; i--)
        for (int j = i; j >= 1; j--)
        {

            if (ans)
            {
                return;
            }
            if (!b[a[i] + a[j]])
            {
                a[now] = a[i] + a[j];
                if (a[now] == n)
                {
                    ans = 1;
                    return;
                }
                b[a[now]] = 1;
                find(now + 1, g);
                b[a[now]] = 0;
            }
        }
}
int main()
{
    cin >> n;
    while (n != 0)
    {
        if (n == 1)
        {
            cout << 1 << '\n';
            cin >> n;
            continue;
        }
        ans = 0;
        int g = 2;
        while (!ans)
        {
            memset(a, 0, sizeof(a));
            b[1] = 1;
            a[1] = 1;
            find(2, g);
            g++;
        }
        sort(a + 1, a + g);
        for (int i = 1; i < g; i++)
        {
            cout << a[i] << ' ';
        }
        cout << '\n';
        cin >> n;
    }
    return 0;
}
```

### 双向搜索

在一些题目中,问题不但具有"初态",还具有明确的"终态",并且从初态开始搜索和从终态开始你想搜索产生的搜索树都能覆盖整个状态空间.在这种情况下,就可以采用双向搜索-----从初态和终态出发各搜索一般状态,产生两棵深度减半的搜索树,在中间交会,组合成最终的答案

避免了层数过深时分支数量的大规模增长

#### 送礼物

感觉,这道题作为双向搜索的例题,就比较特别,这道题是子集和问题的拓展,从给定的N个数中选择几个,使它们的和最接近W.也是一个大体积的背包问题

利用双向搜索的思想,把礼物分成两半,首先,从前一半礼物中选出若干个可能达到0~W之间的所有重量值,存放在一个数组A中,并对数组A进行排序,去重.

然后,进行第二次搜索,尝试从后一半礼物中选出一些.对于每个可能达到的重量值t,在第一部分得到的数组A中二分查找,用二者的和更新答案.

剪枝:

优化搜索顺序:把礼物按照重量降序排序后再分半,搜索

选取适当的"折半划分点":第二次搜索需要在第一次搜索得到的数组中进行二分查找,效率相对较低.

(最后也是,最后一个测试点没有过)

```c++
#include <bits/stdc++.h>
using namespace std;
long long w, n, mid;
long long g[50];
vector<long long> s;
map<long long, bool> mapp;
void add(int l, int r, long long now)
{
    for (int i = l; i <= r; i++)
    {
        if (now + g[i] <= w && !mapp[now + g[i]])
        {
            mapp[now + g[i]] = 1;
            s.push_back(now + g[i]);
            add(i + 1, r, now + g[i]);
        }
    }
}
int main()
{
    cin >> w >> n;
    for (int i = 1; i <= n; i++)
        cin >> g[i];
    s.clear();
    sort(g + 1, g + n + 1, greater<long long>());
    mid = (n / 2) + 2;
    add(1, mid, 0);
    vector<long long> array;
    array.resize(s.size() + 1);
    int q = s.size(), p = q;
    for (int i = 0; i < s.size(); i++)
    {
        array[q--] = s[i];
    }
    sort(array.begin() + 1, array.end());
    array[0] = 0;
    s.clear();
    mapp.clear();
    add(mid + 1, n, 0);
    long long ans = array[p];
    for (int it = 0; it < s.size(); it++)
    {
        int i = lower_bound(array.begin(), array.end(), w - s[it]) - array.begin();
        if (array[i] != w - s[it])
        {
            i--;
            ans = max(ans, array[i] + s[it]);
        }
        else
            ans = max(ans, array[i] + s[it]);
    }
    cout << ans;
    return 0;
}
```

## 广度优先搜索

借助队列来实现广度优先搜索.起初,队列中仅包含起始状态.在广度优先搜索的过程中,我们不断地从队头取出状态,对于该状态面临的所有分支,把沿着每条分支到达的下一个状态插入队尾.重复执行上述过程直到队列为空

#### Bloxorz

典型的"走地图"类问题,即形如"给定一个矩形地图,控制一个物体在地图中按要求移动,求最少步数"的问题.

把变化的部分提取为状态,可以用一个三元组代表一个状态,分别表示,状态和横纵坐标

```c++
#include <bits/stdc++.h>
using namespace std;

struct rec
{
    int x, y, lie;
};                // 状态
char s[510][510]; // 地图
rec st, ed;
int n, m, d[510][510][3]; // 最少步数记录数组
queue<rec> q;
const int dx[4] = {0, 0, -1, 1}, dy[4] = {-1, 1, 0, 0};
bool valid(int x, int y);
void parse_st_ed()
{
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            if (s[i][j] == 'O')
            {
                ed.x = i, ed.y = j, ed.lie = 0, s[i][j] = '.';
            }
            else if (s[i][j] == 'X')
            {
                for (int k = 0; k < 4; k++)
                {
                    int x = i + dx[k], y = j + dy[k];
                    if (valid(x, y) && s[x][y] == 'X')
                    {
                        st.x = min(i, x), st.y = min(j, y);
                        st.lie = k < 2 ? 1 : 2;
                        s[i][j] = s[x][y] = '.';
                        break;
                    }
                }
                if (s[i][j] == 'X')
                    st.x = i, st.y = j, st.lie = 0;
            }
        }
    }
}
bool valid(int x, int y)
{
    return x >= 1 && y >= 1 && x <= n && y <= m;
}
bool valid(rec next)
{
    if (!valid(next.x, next.y))
        return 0;
    if (s[next.x][next.y] == '#')
        return 0;
    if (next.lie == 0 && s[next.x][next.y] != '.')
        return 0;
    if (next.lie == 1 && s[next.x][next.y + 1] == '#')
        return 0;
    if (next.lie == 2 && s[next.x + 1][next.y] == '#')
        return 0;
    return 1;
}
const int next_x[3][4] = {{0, 0, -2, 1}, {0, 0, -1, 1}, {0, 0, -1, 2}};
const int next_y[3][4] = {{-2, 1, 0, 0}, {-1, 2, 0, 0}, {-1, 1, 0, 0}};
const int next_lie[3][4] = {{1, 1, 2, 2}, {0, 0, 1, 1}, {2, 2, 0, 0}};
int bfs()
{
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            for (int k = 0; k < 3; k++)
                d[i][j][k] = -1;
    while (q.size())
        q.pop();
    d[st.x][st.y][st.lie] = 0;
    q.push(st);
    while (q.size())
    {
        rec now = q.front();
        q.pop();
        for (int i = 0; i < 4; i++)
        {
            rec next;
            next.x = now.x + next_x[now.lie][i];
            next.y = now.y + next_y[now.lie][i];
            next.lie = next_lie[now.lie][i];
            if (!valid(next))
                continue;
            if (d[next.x][next.y][next.lie] == -1)
            {
                d[next.x][next.y][next.lie] = d[now.x][now.y][now.lie] + 1;
                q.push(next);
                if (next.x == ed.x && next.y == ed.y && next.lie == ed.lie)
                    return d[next.x][next.y][next.lie];
            }
        }
    }
    return -1;
}
int main()
{
    while (cin >> n >> m && n)
    {
        for (int i = 1; i <= n; i++)
            scanf("%s", s[i] + 1);

        parse_st_ed();
        int ans = bfs();
        if (ans == -1)
            puts("Impossible");
        else
            cout << ans << endl;
    }
}
```

#### 矩阵距离

本题可以看做一道有多个起始状态的flood-fill问题,把矩阵中每个1都看做起点,对每个位置,在从任何一个起点处罚都可以的情况下,求到达该位置所需要的最小步数

在这种具有多个等价的起始状态的问题中,我们只需要在BFS开始之前把这些起始状态全部插入队列.当第一次被访问时,就是最短距离

```c++
#include<bits/stdc++.h>
using namespace std;
const int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
char s[1020][1020];
int d[1020][1020], n, m;
queue<pair<int, int>> q;
int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
        scanf("%s", s[i] + 1);
    memset(d, -1, sizeof(d));
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
            if(s[i][j] == '1')
                q.push(make_pair(i, j)), d[i][j] = 0;
    }
    while(q.size())
    {
        pair<int, int> now = q.front();
        q.pop();
        for (int k = 0; k < 4; k++){
            pair<int, int> next(now.first + dx[k], now.second + dy[k]);
            if(next.first < 1 || next.second < 1 || next.first > n || next.second > m)
                continue;
            if(d[next.first][next.second] == -1){
                d[next.first][next.second] = d[now.first][now.second] + 1;
                q.push(next);
            }
        }
    }
    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= m; j++)
            printf("%d ", d[i][j]);
        puts("");
    }
    return 0;
}
```

#### Pushing Boxes

一道双重BFS算法的问题,因为刚开始的思路错误,导致最后也没写出来(现在看还是感觉,有些复杂,但,从理解上来说,还是可以理解的)

## 广搜变形

### 双端队列BFS

最基本的广度优先搜索,每次沿分支扩展都记为"一步",即在一张边权均为1的图上进行广度优先遍历,求出每个点相对于起点的最短距离.

当遇到图上的边权不仅有1,还有0的情况时,可以采用双端队列,将边权为0的分支直接从队首插入进去,这样可以保证单调性和两段性

#### 电路维修

把电路板上的每个格点看做无向图中的节点,若两点间有线段,则边权为0,否则,边权为1

这道题因为对时间复杂度关键点判断失误,导致if中的条件太多了,然后,最后也没改过来

### 优先队列BFS

对于更加具有普适性的情况,也就是每次扩展都有不同的"代价"时,可以改用优先队列进行广搜,我们可以每次从队列中取出当前代价最小的状态进行扩展,当每个状态第一次从队列中被取出时,就得到了从起始状态到该状态的最小代价.

#### Full Tank

使用二元组记录每个状态,分别记录城市编号和油箱中剩余的汽油量,使用数组存储最少花费

对每个问题,单独进行依次优先队列BFS,起始状态为(S,0).

```c++
#include <bits/stdc++.h>
using namespace std;
struct pii
{
    int first;
    int second;
};

int n, m, q;
int p[1005];
int money[1005][105];
bool jud[1005][105];
map<int, int> expend[10005];
vector<int> v[1005];
int c, s, e;
bool operator<(pair<int, pii> a, pair<int, pii> b)
{
    return a.first > b.first;
}
int zhi;
bool cmp(int a,int b)
{
    return expend[zhi][a] < expend[zhi][b];
}
int main()
{
    cin >> n >> m;
    for (int i = 0; i < n; i++)
        cin >> p[i];
    for (int i = 0; i < m; i++)
    {
        int x, y, z;
        cin >> x >> y >> z;
        expend[x][y] = z;
        expend[y][x] = z; // 是否可能出现重复路线
        v[x].push_back(y);
        v[y].push_back(x);
    }
    for (int i = 0; i < n; i++)
    {
        zhi = i;
        sort(v[i].begin(), v[i].end(),cmp);
    }
    cin >> q;
    while (q--)
    {
        memset(jud, 0, sizeof(jud));
        memset(money, 0x3f, sizeof(money));
        cin >> c >> s >> e;
        priority_queue<pair<int, pii>> pq;
        pq = priority_queue<pair<int, pii>>();
        pq.push({0, {s, 0}});
        money[s][0] = 0;
        while (!pq.empty())
        {
            if (pq.top().second.first == e && pq.top().second.second == 0)
            {
                break;
            }
            pii be = pq.top().second;
            pq.pop();
            if (jud[be.first][be.second])
                continue;
            jud[be.first][be.second] = 1;
            if (be.second < c && (money[be.first][be.second + 1] > money[be.first][be.second] + p[be.first]))
            {
                money[be.first][be.second + 1] = money[be.first][be.second] + p[be.first];
                pq.push({money[be.first][be.second + 1], {be.first, be.second + 1}});
            }
            for (int i = 0; i < v[be.first].size(); i++) // 这里还可以优化一下排序
            {
                if (be.second >= expend[be.first][v[be.first][i]])
                {
                    if ((money[be.first][be.second] < money[v[be.first][i]][be.second - expend[be.first][v[be.first][i]]]))
                    {
                        money[v[be.first][i]][be.second - expend[be.first][v[be.first][i]]] = money[be.first][be.second];
                        pq.push({money[v[be.first][i]][be.second - expend[be.first][v[be.first][i]]], {v[be.first][i], be.second - expend[be.first][v[be.first][i]]}});
                    }
                }
                else
                    break;
            }
        }
        if (!pq.empty())
            cout << money[e][0] << '\n';
        else
            cout << "impossible" << '\n';
    }
}
```

#### Nightmare二

使用双向BFS算法,建立两个队列,分别从男孩和女孩的初始位置开始进行BFS,两遍轮流进行,每轮男孩BFS三层,女孩BFS一层

```c++
#include <bits/stdc++.h>
using namespace std;
int n, m;
char tu[805][805];
pair<int, int> boy, girl, ghost1, ghost2;
int dx1[24] = {-3, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3}, dx2[4] = {1, 0, -1, 0};
int dy1[24] = {0, 1, 0, -1, 2, 1, 0, -1, -2, 3, 2, 1, -1, -2, -3, 2, 1, 0, -1, -2, 1, 0, -1, 0}, dy2[4] = {0, 1, 0, -1};
struct pii
{
    int first;
    int second;
    int timee;
};
bool vaild(int x, int y, int tim)
{
    if (x < 1 || x > m || y < 1 || y > n)
    {
        return false;
    }
    if (tu[x][y] == 'X')
        return false;
    if (abs(x - ghost1.first) + abs(y - ghost1.second) <= 2 * tim || abs(x - ghost2.first) + abs(y - ghost2.second) <= 2 * tim)
        return false;
    return true;
}
int jud[805][805];
int ttime[805][805];
int main()
{
    int T;
    cin >> T;
    while (T--)
    {

        memset(jud, 0, sizeof(jud));
        cin >> n >> m;
        bool jud_g = 0;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j <= m; j++)
            {
                cin >> tu[j][i];
                if (tu[j][i] == 'M')
                    boy.first = j, boy.second = i;
                else if (tu[j][i] == 'G')
                    girl.first = j, girl.second = i;
                else if (tu[j][i] == 'G')
                    girl.first = j, girl.second = i;
                else if (tu[j][i] == 'Z' && !jud_g)
                {
                    ghost1.first = j, ghost1.second = i;
                    jud_g = 1;
                }
                else if (tu[j][i] == 'Z' && jud_g)
                {
                    ghost2.first = j, ghost2.second = i;
                }
            }
        queue<pii> b, g;
        b = queue<pii>();
        g = queue<pii>();
        b.push({boy.first, boy.second, 0});
        g.push({girl.first, girl.second, 0});
        int timee = -1;
        while (!b.empty() || !g.empty())
        {
            if (!b.empty() && !g.empty())
            {
                if (!b.empty() && ((b.front().timee) % 3) ? (b.front().timee / 3 + 1) : (b.front().timee / 3) < g.front().timee)
                {
                    pair<int, int> b1;
                    b1.first = b.front().first, b1.second = b.front().second;
                    if (jud[b1.first][b1.second] == 2)
                    {
                        timee = max(((b.front().timee) % 3) ? (b.front().timee / 3 + 1) : (b.front().timee / 3), ttime[b1.first][b1.second]);
                        break;
                    }
                    int timee2 = b.front().timee + 1;
                    int timee1 = ((b.front().timee + 1) % 3) ? ((b.front().timee + 1) / 3 + 1) : ((b.front().timee + 1) / 3);
                    b.pop();
                    if (jud[b1.first][b1.second] == 1)
                    {
                        continue;
                    }
                    ttime[b1.first][b1.second] = timee1 - 1;
                    jud[b1.first][b1.second] = 1;
                    if (!vaild(b1.first, b1.second, timee1))
                        continue;
                    for (int k = 0; k < 4; k++)
                    {
                        if (vaild(b1.first + dx2[k], b1.second + dy2[k], timee1) && jud[b1.first + dx2[k]][b1.second + dy2[k]] != 1)
                            b.push({b1.first + dx2[k], b1.second + dy2[k], timee2});
                    }
                }
                else if (!g.empty())
                {
                    pair<int, int> g1;
                    g1.first = g.front().first, g1.second = g.front().second;
                    if (jud[g1.first][g1.second] == 1)
                    {
                        timee = max(g.front().timee, ttime[g1.first][g1.second]);
                        break;
                    }
                    int timee1 = g.front().timee + 1;
                    g.pop();
                    if (jud[g1.first][g1.second] == 2)
                    {
                        continue;
                    }
                    ttime[g1.first][g1.second] = timee1 - 1;
                    jud[g1.first][g1.second] = 2;
                    if (!vaild(g1.first, g1.second, timee1))
                        continue;
                    for (int k = 0; k < 4; k++)
                    {
                        if (vaild(g1.first + dx2[k], g1.second + dy2[k], timee1) && jud[g1.first + dx2[k]][g1.second + dy2[k]] != 2)
                            g.push({g1.first + dx2[k], g1.second + dy2[k], timee1});
                    }
                }
            }
            else if (!b.empty())
            {
                pair<int, int> b1;
                b1.first = b.front().first, b1.second = b.front().second;
                if (jud[b1.first][b1.second] == 2)
                {
                    timee = max(((b.front().timee) % 3) ? (b.front().timee / 3 + 1) : (b.front().timee / 3), ttime[b1.first][b1.second]);
                    break;
                }
                int timee2 = b.front().timee + 1;
                int timee1 = ((b.front().timee + 1) % 3) ? ((b.front().timee + 1) / 3 + 1) : ((b.front().timee + 1) / 3);
                b.pop();
                if (jud[b1.first][b1.second] == 1)
                {
                    continue;
                }
                ttime[b1.first][b1.second] = timee1 - 1;
                jud[b1.first][b1.second] = 1;
                if (!vaild(b1.first, b1.second, timee1))
                    continue;
                for (int k = 0; k < 4; k++)
                {
                    if (vaild(b1.first + dx2[k], b1.second + dy2[k], timee1) && jud[b1.first + dx2[k]][b1.second + dy2[k]] != 1)
                        b.push({b1.first + dx2[k], b1.second + dy2[k], timee2});
                }
            }
            else if (!g.empty())
            {
                pair<int, int> g1;
                g1.first = g.front().first, g1.second = g.front().second;
                if (jud[g1.first][g1.second] == 1)
                {
                    timee = max(g.front().timee, ttime[g1.first][g1.second]);
                    break;
                }
                int timee1 = g.front().timee + 1;
                g.pop();
                if (jud[g1.first][g1.second] == 2)
                {
                    continue;
                }
                ttime[g1.first][g1.second] = timee1 - 1;
                jud[g1.first][g1.second] = 2;
                if (!vaild(g1.first, g1.second, timee1))
                    continue;
                for (int k = 0; k < 4; k++)
                {
                    if (vaild(g1.first + dx2[k], g1.second + dy2[k], timee1) && jud[g1.first + dx2[k]][g1.second + dy2[k]] != 2)
                        g.push({g1.first + dx2[k], g1.second + dy2[k], timee1});
                }
            }
        }
        cout << timee << '\n';
    }
}
```

## A*

通过对未来可能产生的代价进行预估(设计一个"估价函数",以任意状态为输入,计算从该状态到目标状态所需代价的估计值).在搜索中,仍然维护一个堆,不断从堆中取出"当前代价+未来估价"最小的状态进行扩展

为保证第一次从堆中取出目标状态时得到的就是最优解,我们设计的估价函数需要满足一个基本准则

![image-20230202113415867](https://gitee.com/yzs1/picture/raw/master/Typora-Images/20230205122529-8.png)

即,估价函数的估值不能大于未来实际代价

这种带有估价函数的优先队列BFS就称为A*算法,估价越接近实际值,算法的效率就越高

#### 第K短路

对于任意正整数i和任意节点x,当第i次从堆中取出包含节点x的二元组时,所对应的dist值就是从S到x的第i短路

把估价函数定为从x到T的最短路长度

```c++
#include <bits/stdc++.h>
using namespace std;
int n, m, s, t, k;
vector<vector<pair<int, int>>> tu;
vector<vector<pair<int, int>>> tu1;
int len[1005];
int jud[1005];
struct pii
{
    int first;
    int second;
};
bool operator<(pii a, pii b)
{
    return a.first > b.first;
}
int main()
{
    cin >> n >> m;
    tu.resize(n + 1);
    tu1.resize(n + 1);
    for (int i = 1; i <= m; i++)
    {
        int a, b, l;
        cin >> a >> b >> l;
        tu1[b].push_back({a, 1});
        tu[a].push_back({b, l});
    }
    cin >> s >> t >> k;
    if (s == t)
        k++;
    priority_queue<pii> q1;
    q1.push({0, t});
    while (!q1.empty())
    {
        pii tt;
        tt = q1.top();
        q1.pop();
        if (jud[tt.second])
            continue;
        else
        {
            jud[tt.second] = 1;
            len[tt.second] = tt.first;
        }
        for (int i = 0; i < tu1[tt.second].size(); i++)
        {
            q1.push({tt.first + tu1[tt.second][i].second, tu1[tt.second][i].first});
        }
    }
    memset(jud, 0, sizeof(jud));
    priority_queue<pii> q2;
    q2.push({len[s], s});
    while (!q2.empty())
    {
        pii tt;
        tt = q2.top();
        if (tt.second == t && jud[tt.second] == k - 1)
            break;
        q2.pop();
        if (jud[tt.second] > k)
            continue;
        else
            jud[tt.second]++;
        for (int i = 0; i < tu[tt.second].size(); i++)
        {
            q2.push({tt.first + tu[tt.second][i].second + len[tu[tt.second][i].first] - len[tt.second], tu[tt.second][i].first});
        }
    }
    if (q2.empty())
        cout << "-1";
    else
        cout << q2.top().first;
    return 0;
}
```

#### 八数码Eight

首先进行可解性判定,求除空格外所有数字排成序列的逆序对数,如果初态和终态的逆序对数奇偶性相同,那么这两个状态互相可达,否则一定不可达

估价函数:从任何一个状态到目标状态的移动步数不可能小于所有数字当前位置与目标位置的曼哈顿距离之和,因此我们可以把估价函数设计为所有数字在state中位置与目标状态end中的位置的曼哈顿距离之和

```c++
#include <bits/stdc++.h>
using namespace std;
map<string, bool> jud;
int a[9], b[9];
int cnt;
bool rig;
int ma[4][4];
struct thr
{
    int first;
    int js;
    string second;
    string third;
};
bool operator<(thr a, thr b)
{
    return a.first + a.js > b.first + b.js;
}
int x[9] = {3, 1, 2, 3, 1, 2, 3, 1, 2};
int y[9] = {3, 1, 1, 1, 2, 2, 2, 3, 3};
int jisuan(string s)
{
    int ans = 0;
    for (int i = 0; i < 9; i++)
    {
        int t = s[i] - '0';
        if (t != 0)
            ans += abs(x[(i + 1) % 9] - x[t]) + abs(y[(i + 1) % 9] - y[t]);
    }
    return ans;
}
bool can(int val)
{
    if (val < 0 || val > 8)
        return false;
    return true;
}
void merge(int l, int mid, int r)
{
    if (l == r)
        return;
    merge(l, (l + mid) / 2, mid);
    merge(mid + 1, (mid + 1 + r) / 2, r);
    // 合并a[l~mid]与a[mid+1~r]
    // a是待排序数组,b是临时数组,cnt是逆序对个数
    int i = l, j = mid + 1;
    for (int k = l; k <= r; k++)
    {
        if (j > r || i <= mid && a[i] <= a[j])
            b[k] = a[i++];
        else
            b[k] = a[j++], cnt += mid - i + 1;
    }
    for (int k = l; k <= r; k++)
        a[k] = b[k];
}
int main()
{
    for (int i = 1; i < 9; i++)
        a[i - 1] = i;
    merge(0, 3, 7);
    rig = cnt % 2;
    string beg = "";
    int num = 0;
    for (int i = 0; i < 9; i++)
    {
        char c;
        cin >> c;
        if (c < '9' && c > '0')
        {
            a[num++] = c - '0';
            beg += c;
        }
        else
        {
            beg += '0';
        }
    }
    cnt = 0;
    merge(0, 3, 7);
    bool rig2 = cnt % 2;
    if (rig != rig2)
    {
        cout << "unsolvable";
        return 0;
    }
    priority_queue<thr> q;
    int chang[4] = {1, 3, -1, -3};
    char cc[4] = {'r', 'd', 'l', 'u'};
    q.push({0, jisuan(beg), beg, ""});
    while (!q.empty())
    {
        thr t;
        t = q.top();
        if (t.second == "123456780")
            break;
        q.pop();
        if (jud[t.second])
            continue;
        jud[t.second] = 1;
        int tt;
        for (int i = 0; i < 9; i++)
            if (t.second[i] == '0')
            {
                tt = i;
                break;
            }
        for (int i = 0; i < 4; i++)
        {
            if (can(tt + chang[i]))
            {
                if ((tt % 3 == 0 && cc[i] == 'l') || (tt % 3 == 2 && cc[i] == 'r'))
                    continue;
                swap(t.second[tt], t.second[tt + chang[i]]);
                q.push({t.first + 1, jisuan(t.second), t.second, t.third + cc[i]});
                swap(t.second[tt], t.second[tt + chang[i]]);
            }
        }
    }
    if (!q.empty())
        cout << q.top().third;
    else
        cout << "unsolvable";
}
```

## IDA*

估价函数与迭代加深的DFS算法的结合

(之后补)
