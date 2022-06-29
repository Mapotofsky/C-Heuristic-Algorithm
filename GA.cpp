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
	// ����ĳ������ֱ�Ϊ �������� ��Ⱥ���� ������� �������
	if (group % 2 == 1) {
		std::cerr << "��Ⱥ����ӦΪ2�ı���" << std::endl;
		return;
	}

	// ����������
	int num = number;                                 // ��������
	std::vector<std::vector<int> > population;        // ��ʼ��Ⱥ
	std::vector<std::vector<int> > cross_father;      // ���游��Ⱥ
	std::vector<std::vector<int> > cross_mother;      // ����ĸ��Ⱥ
	std::vector<std::vector<int> > child_population;  // �Ӵ���Ⱥ
	std::vector<std::vector<int> > mix_population;    // �����Ⱥ
	std::vector<float> fitness1(group);               // ��ʼ��Ⱥ����Ӧ�Ⱥ���
	std::vector<float> reciprocal1(group);            // ��ʼ��Ⱥ��Ӧ�Ⱥ����ĵ���
	std::vector<float> fitness2(group);               // ������Ⱥ����Ӧ�Ⱥ���
	std::vector<float> fitness_mix(2 * group);        // �����Ⱥ����Ӧ�Ⱥ���
	std::vector<float> p_select(group);               // ����ѡ�����
	std::vector<float> fitness_mix_sort(2 * group);   // �����Ⱥ��Ӧ�Ⱥ����������ֵ
	std::vector<float> fitness_mix_copy(2 * group);   // �����Ⱥ��Ӧ�Ⱥ����ĸ���ֵ
	std::vector<int> fmc_index(2 * group);            // �����Ⱥ��Ӧ�Ⱥ������������

	// ����ռ䣬�����ɳ�ʼ�⣬��ʱ��ʼ��δ��������յ㣨0��
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

	// �����ʼ�����Ӧ�Ⱥ���ֵ
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

	// �Ŵ�����
	for (int i = 0; i < generation; ++i) {
		// ѡ��
		// ���ʷ��䷽������Ӧ�ȱ�������
		float sum = accumulate(reciprocal1.begin(), reciprocal1.end(), 0);
		float acc = 0;
		for (int j = 0; j < group; ++j) {
			acc += reciprocal1[j];
			p_select[j] = acc / sum;  // �����ۼƸ���
		}
		// ѡ����巽�������̶�ѡ��
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

		// ����
		int child_num = 0;  // �����Ӵ���

		for (int j = 0; j < group / 2; ++j) {
			float r = rand() / float(RAND_MAX);
			if (r < p_cross) {
				// ����ǿ�Ƚ�С��һ�㽻��
				int po11 = rand() % plen;
				int po21 = rand() % plen;
				int po12, po22;
				int cvalue1 = cross_father[j][po11];
				int cvalue2 = cross_mother[j][po21];
				// ����Ⱥ cross_father �еĵ� j ������ĵ� po11 ��ֵ cvalue1
				// ����Ⱥ cross_mother �еĵ� j ������ĵ� po21 ��ֵ���� cvalue2
				// ͬʱ��֤���ظ�����
				// ��Ⱥ cross_father �� j ������ �� cvalue2 ����λ�� po12
				// ����Ⱥ cross_mother �� j ������ �� cvalue1 ����λ�� po22 ��Ӧ��ֵҲҪ����
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

		// ����
		for (int j = 0; j < child_num; ++j) {
			float r = rand() / float(RAND_MAX);
			if (r < p_mutate) {
				std::vector<int> path1(plen);
				std::copy(child_population[j].begin(), child_population[j].end(), path1.begin());
				// ����ǿ�Ƚϴ����λ�͵���
				if (r < 0.5) {            // ��λ��
					path1.clear();
					int sort_c[3] = { rand() % plen, rand() % plen, rand() % plen };
					std::sort(sort_c, sort_c + 3);
					// ���λ���
					copy(child_population[j].begin(), child_population[j].begin() + sort_c[0], std::back_inserter(path1));
					copy(child_population[j].begin() + sort_c[1], child_population[j].begin() + sort_c[2], std::back_inserter(path1));
					copy(child_population[j].begin() + sort_c[0], child_population[j].begin() + sort_c[1], std::back_inserter(path1));
					copy(child_population[j].begin() + sort_c[2], child_population[j].end(), std::back_inserter(path1));

					copy(path1.begin(), path1.end(), child_population[j].begin());
				}
				else {                    // ���÷�
					path1.clear();
					int c1 = rand() % plen;
					int c2 = rand() % plen;
					if (c1 > c2) {
						std::swap(c1, c2);
					}
					// ��ȡ����ת
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

		// �����Ӵ���Ⱥ����Ӧ�Ⱥ���
		float temp = 0;
		for (int j = 0; j < child_num; ++j) {
			temp = 0;
			for (int k = 0; k < plen - 1; ++k) {
				temp += mat(child_population[j][k], child_population[j][k + 1]);
			}
			temp += mat(0, child_population[j][0]) + mat(child_population[j][plen - 1], 0);
			fitness2[j] = temp;
		}

		// �����Ⱥ
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

		// ����Ӧ������
		copy(fitness_mix.begin(), fitness_mix.end(), fitness_mix_copy.begin());
		copy(fitness_mix.begin(), fitness_mix.end(), fitness_mix_sort.begin());
		sort(fitness_mix_sort.begin(), fitness_mix_sort.end());
		for (int j = 0; j < group + child_num; ++j) {
			// ��С�����Ҷ�Ӧ�±�
			fmc_index[j] = find(fitness_mix_copy.begin(), fitness_mix_copy.end(), fitness_mix_sort[j]) - fitness_mix_copy.begin();
			fitness_mix_copy[fmc_index[j]] = INFINITY;
		}

		// �����µ���Ⱥ����Ӧ��
		for (int j = 0; j < child_num; ++j) {
			copy(mix_population[fmc_index[j]].begin(), mix_population[fmc_index[j]].end(), population[j].begin());
			fitness1[j] = fitness_mix_sort[j];
			reciprocal1[j] = 1 / fitness1[j];
		}
		// ������
		obj = fitness1[0];
		nsolution = population[0];
		nsolution.emplace(nsolution.begin(), 0);
		nsolution.emplace(nsolution.end(), 0);
	}
}

int main()
{
	srand(time(0));  // �������������
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