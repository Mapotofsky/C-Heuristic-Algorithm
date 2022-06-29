#include "header.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <algorithm>
#include <ctime>
using namespace Eigen;

void TSP::aco(int max_G, int ants, float alpha, float beta, float tho, float Q)
{
	// 输入的超参数分别为 最大迭代次数 信息素启发因子 期望值启发因子 信息素挥发度 信息素增加强度
	
	// 参数及变量
	int num = number;
	int plen = num - 1;
	std::vector<std::vector<int> > Table;   // 禁忌表，存储走过的路径
	std::vector<int> tabu;                  // 记录临时路径
	std::vector<std::vector<float> > eta;   // 能见度
	std::vector<std::vector<float> > tau;   // 信息素
	std::vector<std::vector<float> > dtau;  // 信息素变化量
	std::vector<std::vector<int> > R_best;  // 每代最短路径
	std::vector<float> L_best(max_G);       // 每代最短路径长度
	std::vector<int> visited;               // 已访问的城市
	std::vector<int> allow;                 // 待访问的城市
	std::vector<float> P;                   // 转移概率
	std::vector<float> P_temp;              // 计算转移概率的中间值
	std::vector<float> L_ants(ants);        // 每只蚂蚁路径长度

	// 初始化变量
	Table.resize(ants);
	for (int i = 0; i < ants; ++i) {
		Table[i].resize(plen);
	}
	eta.resize(num);
	for (int i = 0; i < num; ++i) {
		eta[i].resize(num);
		for (int j = 0; j < num; ++j) {
			eta[i][j] = 1 / mat(i, j);
		}
	}
	tau.resize(num);
	for (int i = 0; i < num; ++i) {
		tau[i].resize(num);
		for (int j = 0; j < num; ++j) {
			tau[i][j] = 1;
		}
	}
	dtau.resize(num);
	for (int i = 0; i < num; ++i) {
		dtau[i].resize(num);
	}
	R_best.resize(max_G);
	for (int i = 0; i < max_G; ++i) {
		R_best[i].resize(plen);
	}
	visited.resize(plen);
	allow.resize(plen);
	P.resize(plen);
	P_temp.resize(plen);

	// 生成蚁群
	for (int i = 0; i < max_G; ++i) {
		// 生成起点
		std::vector<int> start = linspace(num);
		for (int j = 0; j < ants; ++j) {
			int r = rand() % plen;
			Table[j][0] = start[r];
		}
		
		// 逐个城市路径选择
		for (int j = 1; j < plen; ++j) {
			// 逐个蚂蚁路径选择
			for (int k = 0; k < ants; ++k) {
				visited.clear();  // 已访问城市
				allow.clear();    // 未访问城市
				P.clear();        // 转移概率
				P_temp.clear();   // 计算转移概率的中间值
				float P_sum = 0;
				float P_tempsum = 0;

				// 添加已访问城市
				for (int v = 0; v < j; ++v) {
					visited.emplace_back(Table[k][v]);
				}
				// 添加未访问城市
				for (int v = 0; v < plen; ++v) {
					//若在visited中没有找到某节点，则说明还未被访问
					if (find(visited.begin(), visited.end(), start[v]) == visited.end()) {
						allow.emplace_back(start[v]);
						P.emplace_back(0.0);
						P_temp.emplace_back(0.0);
					}
				}

				// 计算转移概率
				for (int v = 0; v < P_temp.size(); ++v) {
					// tau[visited.back()][allow[v]]即当前所在城市到第v个未访问城市的信息素
					P_temp[v] = std::pow(tau[visited.back()][allow[v]], alpha) * std::pow(eta[visited.back()][allow[v]], beta);
					P_tempsum += P_temp[v];
				}
				for (int v = 0; v < P.size(); ++v) {
					P[v] = P_temp[v] / P_tempsum;
					P_sum += P[v];
				}

				// 轮盘赌选择下一个城市
				float r = (rand() / (float)RAND_MAX) * P_sum;
				float choose = 0;
				for (int v = 0; v < P.size(); ++v) {
					choose += P[v];
					if (r < choose) {
						Table[k][j] = allow[v];
						break;
					}
				}
			}
		}

		// 计算本次迭代结果
		// 每只蚂蚁走过路程
		for (int j = 0; j < ants; ++j) {
			for (int k = 0; k < plen - 1; ++k) {
				L_ants[j] += mat(Table[j][k], Table[j][k + 1]);
			}
			L_ants[j] += mat(0, Table[j][0]) + mat(Table[j][plen - 1], 0);
		}
		// 寻找本代最小路程
		float L_min = L_ants[0];
		int min_index = 0;
		for (int j = 0; j < ants; ++j) {
			if (L_ants[j] < L_min) {
				L_min = L_ants[j];
				min_index = j;
			}
		}
		L_best[i] = L_min;
		std::copy(Table[min_index].begin(), Table[min_index].end(), R_best[i].begin());

		// 信息素浓度更新规则：蚂蚁圈系统
		for (int j = 0; j < ants; ++j) {
			dtau[0][Table[j][0]] += Q / L_ants[j];
			for (int k = 0; k < plen - 1; ++k) {
				dtau[Table[j][k]][Table[j][k + 1]] += Q / L_ants[j];
			}
			dtau[Table[j][plen - 1]][0] += Q / L_ants[j];
		}

		// 信息素浓度挥发与叠加
		for (int j = 0; j < num; ++j) {
			for (int k = 0; k < num; ++k) {
				tau[j][k] = tau[j][k] * tho + dtau[j][k];
			}
		}

		// 各表归零
		for (int j = 0; j < ants; ++j) {
			for (int k = 0; k < plen; ++k) {
				Table[j][k] = 0;
			}
			L_ants[j] = 0;
		}
	}

	// 保存结果
	int min_index = std::min_element(L_best.begin(), L_best.end()) - L_best.begin();
	obj = L_best[min_index];
	nsolution = R_best[min_index];
	nsolution.emplace(nsolution.begin(), 0);
	nsolution.emplace(nsolution.end(), 0);
}

int main()
{
	srand(time(0));  // 重置随机数种子
	Matrix<float, Dynamic, 2> a;
	a.resize(10, 2);
	a << 1, 1,
		0, 0,
		2, 3,
		4, 4,
		5, 6,
		7, 5,
		5, 3,
		9, 9,
		1, 3,
		6, 8;
	data city(a);
	TSP solver(city);

	clock_t start, end;
	start = clock();
	solver.aco();
	end = clock();
	std::cout << "time = " << double(end - start) / CLOCKS_PER_SEC << 's' << std::endl;

	std::cout << solver.showobj() << std::endl;
	std::vector<int> p = solver.showsol();
	for (int i = 0; i < p.size(); ++i) {
		std::cout << p[i] << ' ';
	}
	std::cout << std::endl;
}