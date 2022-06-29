#include "header.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <algorithm>
#include <ctime>
#include <numeric>
using namespace Eigen;

void TSP::ga(int generation, int group, float p_cross, float p_mutate)
{
	// 输入的超参数分别为 进化代数 种群数量 交叉概率 变异概率
	if (group % 2 == 1) {
		std::cerr << "种群数量应为2的倍数" << std::endl;
		return;
	}

	// 参数及变量
	int num = number;                                 // 城市数量
	std::vector<std::vector<int> > population;        // 初始种群
	std::vector<std::vector<int> > cross_father;      // 交叉父种群
	std::vector<std::vector<int> > cross_mother;      // 交叉母种群
	std::vector<std::vector<int> > child_population;  // 子代种群
	std::vector<std::vector<int> > mix_population;    // 混合种群
	std::vector<float> fitness1(group);               // 初始种群的适应度函数
	std::vector<float> reciprocal1(group);            // 初始种群适应度函数的倒数
	std::vector<float> fitness2(group);               // 变异种群的适应度函数
	std::vector<float> fitness_mix(2 * group);        // 混合种群的适应度函数
	std::vector<float> p_select(group);               // 个体选择概率
	std::vector<float> fitness_mix_sort(2 * group);   // 混合种群适应度函数的排序后值
	std::vector<float> fitness_mix_copy(2 * group);   // 混合种群适应度函数的复制值
	std::vector<int> fmc_index(2 * group);            // 混合种群适应度函数排序后索引

	// 申请空间，并生成初始解，此时初始解未加上起点终点（0）
	std::vector<int> path0 = linspace(num);
	int plen = path0.size();
	population.resize(group);
	for (int i = 0; i < group; ++i) {
		population[i].resize(plen);
		std::copy(path0.begin(), path0.end(), population[i].begin());
		random_shuffle(population[i].begin(), population[i].end());
	}
	cross_father = population;
	cross_mother = population;
	child_population = population;

	mix_population.resize(2 * group);
	for (int i = 0; i < 2 * group; ++i) {
		mix_population[i].resize(plen);
	}

	// 计算初始解的适应度函数值
	float temp = 0;
	for (int i = 0; i < group; ++i) {
		temp = 0;
		for (int j = 0; j < plen - 1; ++j) {
			temp += mat(population[i][j], population[i][j + 1]);
		}
		temp += mat(0, population[i][0]) + mat(population[i][plen - 1], 0);
		fitness1[i] = temp;
		reciprocal1[i] = 1 / fitness1[i];
	}

	// 遗传进化
	for (int i = 0; i < generation; ++i) {
		// 选择
		// 概率分配方法：适应度比例方法
		float sum = accumulate(reciprocal1.begin(), reciprocal1.end(), 0);
		float acc = 0;
		for (int j = 0; j < group; ++j) {
			acc += reciprocal1[j];
			p_select[j] = acc / sum;  // 计算累计概率
		}
		// 选择个体方法：轮盘赌选择
		for (int j = 0; j < group / 2; ++j) {
			float r1 = rand() / float(RAND_MAX);
			float r2 = rand() / float(RAND_MAX);
			for (int k = 0; k < group; ++k) {
				if (r1 < p_select[k] && (k == 0 || r1 >= p_select[k - 1])) {
					std::copy(population[k].begin(), population[k].end(), cross_father[j].begin());
				}
				if (r2 < p_select[k] && (k == 0 || r2 >= p_select[k - 1])) {
					std::copy(population[k].begin(), population[k].end(), cross_mother[j].begin());
				}
			}
		}

		// 交叉
		int child_num = 0;  // 交叉子代数

		for (int j = 0; j < group / 2; ++j) {
			float r = rand() / float(RAND_MAX);
			if (r < p_cross) {
				// 采用强度较小的一点交叉
				int po11 = rand() % plen;
				int po21 = rand() % plen;
				int po12, po22;
				int cvalue1 = cross_father[j][po11];
				int cvalue2 = cross_mother[j][po21];
				// 将种群 cross_father 中的第 j 个个体的第 po11 个值 cvalue1
				// 与种群 cross_mother 中的第 j 个个体的第 po21 个值交换 cvalue2
				// 同时保证无重复，即
				// 种群 cross_father 第 j 个个体 中 cvalue2 所在位置 po12
				// 与种群 cross_mother 第 j 个个体 中 cvalue1 所在位置 po22 对应的值也要交换
				std::swap(cross_father[j][po11], cross_mother[j][po21]);
				for (int m = 0; m < plen; ++m) {
					if (cross_father[j][m] == cvalue2) {
						po12 = m;
						break;
					}
				}
				for (int m = 0; m < plen; ++m) {
					if (cross_mother[j][m] == cvalue1) {
						po22 = m;
						break;
					}
				}
				std::swap(cross_father[j][po12], cross_mother[j][po22]);

				copy(cross_father[j].begin(), cross_father[j].end(), child_population[child_num++].begin());
				copy(cross_mother[j].begin(), cross_mother[j].end(), child_population[child_num++].begin());
			}
		}

		// 变异
		for (int j = 0; j < child_num; ++j) {
			float r = rand() / float(RAND_MAX);
			if (r < p_mutate) {
				std::vector<int> path1(plen);
				std::copy(child_population[j].begin(), child_population[j].end(), path1.begin());
				// 采用强度较大的移位和倒置
				if (r < 0.5) {            // 移位法
					path1.clear();
					int sort_c[3] = { rand() % plen, rand() % plen, rand() % plen };
					std::sort(sort_c, sort_c + 3);
					// 两段互换
					copy(child_population[j].begin(), child_population[j].begin() + sort_c[0], std::back_inserter(path1));
					copy(child_population[j].begin() + sort_c[1], child_population[j].begin() + sort_c[2], std::back_inserter(path1));
					copy(child_population[j].begin() + sort_c[0], child_population[j].begin() + sort_c[1], std::back_inserter(path1));
					copy(child_population[j].begin() + sort_c[2], child_population[j].end(), std::back_inserter(path1));

					copy(path1.begin(), path1.end(), child_population[j].begin());
				}
				else {                    // 倒置法
					path1.clear();
					int c1 = rand() % plen;
					int c2 = rand() % plen;
					if (c1 > c2) {
						std::swap(c1, c2);
					}
					// 截取并反转
					std::vector<int> path_tmp(child_population[j].begin() + c1, child_population[j].begin() + c2);
					int half_tmp = (path_tmp.size() - 1) / 2;
					int n = path_tmp.size() - 1;
					for (; n > half_tmp; --n) {
						std::swap(path_tmp[n], path_tmp[path_tmp.size() - n - 1]);
					}

					copy(child_population[j].begin(), child_population[j].begin() + c1, std::back_inserter(path1));
					copy(path_tmp.begin(), path_tmp.end(), std::back_inserter(path1));
					copy(child_population[j].begin() + c2, child_population[j].end(), std::back_inserter(path1));

					copy(path1.begin(), path1.end(), child_population[j].begin());
				}
			}
		}

		// 计算子代种群的适应度函数
		float temp = 0;
		for (int j = 0; j < child_num; ++j) {
			temp = 0;
			for (int k = 0; k < plen - 1; ++k) {
				temp += mat(child_population[j][k], child_population[j][k + 1]);
			}
			temp += mat(0, child_population[j][0]) + mat(child_population[j][plen - 1], 0);
			fitness2[j] = temp;
		}

		// 混合种群
		mix_population.resize(group + child_num);
		fitness_mix.resize(group + child_num);
		fitness_mix_copy.resize(group + child_num);
		fitness_mix_sort.resize(group + child_num);
		fmc_index.resize(group + child_num);
		for (int j = 0; j < group + child_num; ++j) {
			if (j < group) {
				mix_population[j] = population[j];
			}
			else {
				mix_population[j] = child_population[j - group];
			}
		}
		copy(fitness1.begin(), fitness1.end(), fitness_mix.begin());
		copy(fitness2.begin(), fitness2.begin() + child_num, fitness_mix.begin() + group);

		// 按适应度排序
		copy(fitness_mix.begin(), fitness_mix.end(), fitness_mix_copy.begin());
		copy(fitness_mix.begin(), fitness_mix.end(), fitness_mix_sort.begin());
		sort(fitness_mix_sort.begin(), fitness_mix_sort.end());
		for (int j = 0; j < group + child_num; ++j) {
			// 从小到大找对应下标
			fmc_index[j] = find(fitness_mix_copy.begin(), fitness_mix_copy.end(), fitness_mix_sort[j]) - fitness_mix_copy.begin();
			fitness_mix_copy[fmc_index[j]] = INFINITY;
		}

		// 保存新的种群和适应度
		for (int j = 0; j < child_num; ++j) {
			copy(mix_population[fmc_index[j]].begin(), mix_population[fmc_index[j]].end(), population[j].begin());
			fitness1[j] = fitness_mix_sort[j];
			reciprocal1[j] = 1 / fitness1[j];
		}
		// 保存结果
		obj = fitness1[0];
		nsolution = population[0];
		nsolution.emplace(nsolution.begin(), 0);
		nsolution.emplace(nsolution.end(), 0);
	}
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
	solver.ga();
	end = clock();
	std::cout << "time = " << double(end - start) / CLOCKS_PER_SEC << 's' << std::endl;

	std::cout << solver.showobj() << std::endl;
	std::vector<int> p = solver.showsol();
	for (int i = 0; i < p.size(); ++i) {
		std::cout << p[i] << ' ';
	}
	std::cout << std::endl;
}