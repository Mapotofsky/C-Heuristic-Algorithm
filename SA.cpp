#include "header.h"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <algorithm>
#include <ctime>
using namespace Eigen;

void TSP::sa(float t0, int max_G, int max_perT, float cooling)
{
	// ����ĳ������ֱ�Ϊ ��ʼ�¶� ���������� ÿ���¶��µĵ������� ����ϵ��
	
    // ����������
	int num = number;            // ��������
	float T = t0;                // ��ǰ�¶�
	float result0 = 0;           // ÿ���¶��µ����Ž��
	float result = 0;            // ÿ����Ľ��
	float p1 = 0.33, p2 = 0.33;  // ���Ŷ���������
	float r = 0;                 // ����ѡ���Ŷ��������Ƿ�����½�������

	// ���ɳ�ʼ�⣬��ʱ��ʼ��δ��������յ㣨0��
	std::vector<int> path0 = linspace(num);
	random_shuffle(path0.begin(), path0.end());
	int plen = path0.size();

	// �����ʼ���Ŀ�꺯��ֵ
	float temp = 0;
	for (int i = 0; i < plen - 1; ++i) {
		temp += mat(path0[i], path0[i + 1]);
	}
	temp += mat(0, path0[0]) + mat(path0[plen - 1], 0);
	result0 = temp;
	result = temp;

	// ģ���˻�
	for (int i = 0; i < max_G; ++i) {
		for (int j = 0; j < max_perT; ++j) {
			// ������·��
			r = rand() / float(RAND_MAX);  // 0��1���������

			std::vector<int> path1(plen);
			std::copy(path0.begin(), path0.end(), path1.begin());  // copy�ȸ�ֵ����Ч
			if (r < p1) {            // ������
				std::swap(path1[rand() % plen], path1[rand() % plen]);
			}
			else if (r < p1 + p2) {  // ��λ��
				path1.clear();
				int sort_c[3] = { rand() % plen, rand() % plen, rand() % plen };
				std::sort(sort_c, sort_c + 3);
				// ���λ���
				copy(path0.begin(), path0.begin() + sort_c[0], std::back_inserter(path1));
				copy(path0.begin() + sort_c[1], path0.begin() + sort_c[2], std::back_inserter(path1));
				copy(path0.begin() + sort_c[0], path0.begin() + sort_c[1], std::back_inserter(path1));
				copy(path0.begin() + sort_c[2], path0.end(), std::back_inserter(path1));
			}
			else {                   // ���÷�
				path1.clear();
				int c1 = rand() % plen;
				int c2 = rand() % plen;
				if (c1 > c2) {
					std::swap(c1, c2);
				}
				// ��ȡ����ת
				std::vector<int> path_tmp(path0.begin() + c1, path0.begin() + c2);
				int half_tmp = (path_tmp.size() - 1) / 2;
				int n = path_tmp.size() - 1;
				for (; n > half_tmp; --n) {
					std::swap(path_tmp[n], path_tmp[path_tmp.size() - n - 1]);
				}

				copy(path0.begin(), path0.begin() + c1, std::back_inserter(path1));
				copy(path_tmp.begin(), path_tmp.end(), std::back_inserter(path1));
				copy(path0.begin() + c2, path0.end(), std::back_inserter(path1));
			}

			// ����Ŀ�꺯��
			temp = 0;
			for (int iter = 0; iter < plen - 1; ++iter) {
				temp += mat(path0[iter], path0[iter + 1]);
			}
			temp += mat(0, path0[0]) + mat(path0[plen - 1], 0);
			result = temp;

			if (result < result0) {  // �½����̣�ֱ�ӽ���
				copy(path1.begin(), path1.end(), path0.begin());
				result0 = result;
			}
			else {                   // metropolis׼��
				float p = exp(-(result - result0) / T);
				r = rand() / float(RAND_MAX);
				if (r < p) {
					copy(path1.begin(), path1.end(), path0.begin());
					result0 = result;
				}
			}

			// �ж��Ƿ�������Ž�
			if (result0 < obj) {
				obj = result0;
				std::vector<int> path = path0;
				path.emplace(path.begin(), 0);
				path.emplace(path.end(), 0);
				nsolution = path;
			}
		}
		T *= cooling;
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
	solver.sa();
	end = clock();
	std::cout << "time = " << double(end - start) / CLOCKS_PER_SEC << 's' << std::endl;

	std::cout << solver.showobj() << std::endl;
	std::vector<int> p = solver.showsol();
	for (int i = 0; i < p.size(); ++i) {
		std::cout << p[i] << ' ';
	}
	std::cout << std::endl;
}