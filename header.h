#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include<math.h>
using namespace Eigen;
const float max = 9999999.0;
std::vector<int> linspace(int n) {//���ɲ�Ϊ1�ĵȲ�����
	std::vector<int> a;
	for (int i = 1; i < n; i++) a.push_back(i);
	return a;
}
class data {//�������������̵ĳ�������
public:
	Matrix<float, Dynamic, 2> xy;
	int number = xy.cols();
	data(Matrix<float, Dynamic, 2>);
	data() :number(0) {};
	Matrix<float, Dynamic, Dynamic> cal();
};
data::data(Matrix<float, Dynamic, 2> a) {
	xy = a;
	number = a.rows();
}

Matrix<float, Dynamic, Dynamic> data::cal() {
	Matrix<float, Dynamic, Dynamic> mat;
	mat.resize(number, number);
	for (int i = 0; i < number; i++) {
		for (int j = 0; j < number; j++) {
			mat(i, j) = sqrt((xy(i, 0) - xy(j, 0)) * (xy(i, 0) - xy(j, 0)) + (xy(i, 1) - xy(j, 1)) * (xy(i, 1) - xy(j, 1)));
		}
	}
	return mat;
}
class TSP {//���������̵����
	friend std::vector<int> linspace(int n);
private://city�洢������Ϣ��goalΪĿ��⣬scobjΪ�ɳ�����⣬nsolutionΪ��ǰ�⣬objΪ��ǰĿ�꺯��ֵ��statusΪ���״̬
	data city;  // ������Ϣ
	int number;
	float goal;  // Ŀ���
	float scobj;  // �ɳ������
	Matrix<float, Dynamic, Dynamic> mat;
	std::vector<int> nsolution;// ��ǰ�⣬��Ӧ���Ǹ�n+1ά������,һ������µ�һλΪ0����0-2-3-1-0����ʾ��0�ų��е�2���ٵ�3�ŵ�1�����ص�0�ţ�
	float obj;  // ��ǰĿ�꺯��ֵ
	std::string status;  // ���״̬
public://�ĸ���ʼ������
	TSP(data a, float mubiao = max);
	TSP(Matrix<float, Dynamic, Dynamic> mat, float mubiao = max);
	TSP(Matrix<float, Dynamic, Dynamic> mat, std::vector<int> solution, float mubiao = max);
	TSP(data a, std::vector<int> solution, float mubiao = max);
	float calu() const;
	float showobj() const { return obj; }
	std::vector<int> showsol() const { return nsolution; }
	int shownum() const { return number; }
	//������Ϊ������ɷ�������ز�����δ����
	//ǰ4��Ϊ��ʼ������ɷ������Ǳ�Ҫ�����ֱ�Ϊ̰�ģ��Ľ�̰�ģ�LKH�㷨���������㷨��������Ϊ��ȷ��⣬�ֱ�Ϊ��֧���硢��̬�滮�������㷨,�ٺ�����Ϊ����ʽ�㷨
	//�ֱ�Ϊ��Ⱥ�㷨,�˻��㷨,�Ŵ��㷨�����һ��Ϊǿ��ѧϰ�㷨�������о��ɹ�����
	void tc();
	void ptc();
	void lkh();
	void hsc();
	void b_and_b();
	void dp();
	void backtrack(std::vector<int> solution = { 0 }, float nobj = 0, std::vector<int> selection = { 0 });
	void aco(int max_G = 500, int ants = 100, float alpha = 1, float beta = 5, float tho = 0.9, float Q = 100);
	void sa(float t0 = 1000, int max_G = 500, int max_perT = 100, float cooling = 0.98);
	void ga(int generation = 1000, int group = 50, float p_cross = 0.7, float p_mutate = 0.001);
	void dl();
};
float TSP::calu() const {
	float zhi = 0;
	for (int i = 0; i < number; i++) {
		zhi += mat(nsolution[i], nsolution[i + 1]);
	}
	return zhi;
}
TSP::TSP(data a, float mubiao) {
	city = a;
	number = a.number;
	mat = a.cal();
	goal = mubiao;
	obj = max;
}
TSP::TSP(Matrix<float, Dynamic, Dynamic> gmat, float mubiao) {
	if (mat.rows() != mat.cols()) {
		std::cerr << "�����ڽӾ���" << std::endl;
		exit(1);
	}
	number = gmat.rows();
	mat = gmat;
	goal = mubiao;
	obj = max;
}
TSP::TSP(Matrix<float, Dynamic, Dynamic> gmat, std::vector<int> solution, float mubiao) {
	if (mat.rows() != mat.cols()) {
		std::cerr << "�����ڽӾ���" << std::endl;
		exit(1);
	}
	number = gmat.rows();
	mat = gmat;
	goal = mubiao;
	nsolution = solution;
	obj = calu();
}
TSP::TSP(data a, std::vector<int> solution, float mubiao) {
	city = a;
	number = a.number;
	mat = a.cal();
	goal = mubiao;
	nsolution = solution;
	obj = calu();
}
void TSP::backtrack(std::vector<int> a, float nobj, std::vector<int> selection) {
	int num = a.size();
	if (num == 1) selection = linspace(number);
	if (num == number) {
		nobj += mat(a[num - 1], a[num - 2]) + mat(a[num - 1], 0);
		if (nobj < obj) {
			obj = nobj;
			nsolution = a;
			nsolution.push_back(0);
		}

		return;
	}
	if (nobj > obj) return;
	std::vector<int> b = a;
	std::vector<int> selections = selection;
	for (int i = 0; i < selection.size(); i++) {
		b.push_back(selection[i]);
		selections.erase(selections.begin() + i);
		backtrack(b, nobj + mat(b[num], b[num - 1]), selections);
		b.pop_back();
		selections.insert(selections.begin() + i, i + 1);
	}
}
class MTSP {//��������ĸ�࣬�ɲ�ͬԼ���������ɲ�ͬ����
private:
	data city;
	float scobj;
	Matrix<int, 1, Dynamic> nsolution;
	float obj;
	bool status;
public:
	MTSP();
};
class MTSP_n :MTSP {//�����������Ƶ�
private:
	int maxn;
};
//�ȵ�,��������Լ������