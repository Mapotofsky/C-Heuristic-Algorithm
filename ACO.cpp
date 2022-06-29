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
	// ����ĳ������ֱ�Ϊ ���������� ��Ϣ���������� ����ֵ�������� ��Ϣ�ػӷ��� ��Ϣ������ǿ��
	
	// ����������
	int num = number;
	int plen = num - 1;
	std::vector<std::vector<int> > Table;   // ���ɱ��洢�߹���·��
	std::vector<int> tabu;                  // ��¼��ʱ·��
	std::vector<std::vector<float> > eta;   // �ܼ���
	std::vector<std::vector<float> > tau;   // ��Ϣ��
	std::vector<std::vector<float> > dtau;  // ��Ϣ�ر仯��
	std::vector<std::vector<int> > R_best;  // ÿ�����·��
	std::vector<float> L_best(max_G);       // ÿ�����·������
	std::vector<int> visited;               // �ѷ��ʵĳ���
	std::vector<int> allow;                 // �����ʵĳ���
	std::vector<float> P;                   // ת�Ƹ���
	std::vector<float> P_temp;              // ����ת�Ƹ��ʵ��м�ֵ
	std::vector<float> L_ants(ants);        // ÿֻ����·������

	// ��ʼ������
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

	// ������Ⱥ
	for (int i = 0; i < max_G; ++i) {
		// �������
		std::vector<int> start = linspace(num);
		for (int j = 0; j < ants; ++j) {
			int r = rand() % plen;
			Table[j][0] = start[r];
		}
		
		// �������·��ѡ��
		for (int j = 1; j < plen; ++j) {
			// �������·��ѡ��
			for (int k = 0; k < ants; ++k) {
				visited.clear();  // �ѷ��ʳ���
				allow.clear();    // δ���ʳ���
				P.clear();        // ת�Ƹ���
				P_temp.clear();   // ����ת�Ƹ��ʵ��м�ֵ
				float P_sum = 0;
				float P_tempsum = 0;

				// ����ѷ��ʳ���
				for (int v = 0; v < j; ++v) {
					visited.emplace_back(Table[k][v]);
				}
				// ���δ���ʳ���
				for (int v = 0; v < plen; ++v) {
					//����visited��û���ҵ�ĳ�ڵ㣬��˵����δ������
					if (find(visited.begin(), visited.end(), start[v]) == visited.end()) {
						allow.emplace_back(start[v]);
						P.emplace_back(0.0);
						P_temp.emplace_back(0.0);
					}
				}

				// ����ת�Ƹ���
				for (int v = 0; v < P_temp.size(); ++v) {
					// tau[visited.back()][allow[v]]����ǰ���ڳ��е���v��δ���ʳ��е���Ϣ��
					P_temp[v] = std::pow(tau[visited.back()][allow[v]], alpha) * std::pow(eta[visited.back()][allow[v]], beta);
					P_tempsum += P_temp[v];
				}
				for (int v = 0; v < P.size(); ++v) {
					P[v] = P_temp[v] / P_tempsum;
					P_sum += P[v];
				}

				// ���̶�ѡ����һ������
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

		// ���㱾�ε������
		// ÿֻ�����߹�·��
		for (int j = 0; j < ants; ++j) {
			for (int k = 0; k < plen - 1; ++k) {
				L_ants[j] += mat(Table[j][k], Table[j][k + 1]);
			}
			L_ants[j] += mat(0, Table[j][0]) + mat(Table[j][plen - 1], 0);
		}
		// Ѱ�ұ�����С·��
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

		// ��Ϣ��Ũ�ȸ��¹�������Ȧϵͳ
		for (int j = 0; j < ants; ++j) {
			dtau[0][Table[j][0]] += Q / L_ants[j];
			for (int k = 0; k < plen - 1; ++k) {
				dtau[Table[j][k]][Table[j][k + 1]] += Q / L_ants[j];
			}
			dtau[Table[j][plen - 1]][0] += Q / L_ants[j];
		}

		// ��Ϣ��Ũ�Ȼӷ������
		for (int j = 0; j < num; ++j) {
			for (int k = 0; k < num; ++k) {
				tau[j][k] = tau[j][k] * tho + dtau[j][k];
			}
		}

		// �������
		for (int j = 0; j < ants; ++j) {
			for (int k = 0; k < plen; ++k) {
				Table[j][k] = 0;
			}
			L_ants[j] = 0;
		}
	}

	// ������
	int min_index = std::min_element(L_best.begin(), L_best.end()) - L_best.begin();
	obj = L_best[min_index];
	nsolution = R_best[min_index];
	nsolution.emplace(nsolution.begin(), 0);
	nsolution.emplace(nsolution.end(), 0);
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