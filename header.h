#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include<math.h>
using namespace Eigen;
const float max = 9999999.0;
std::vector<int> linspace(int n) {//生成差为1的等差数列
	std::vector<int> a;
	for (int i = 1; i < n; i++) a.push_back(i);
	return a;
}
class data {//用来保存旅行商的城市数据
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
class TSP {//单人旅行商的求解
	friend std::vector<int> linspace(int n);
private://city存储城市信息，goal为目标解，scobj为松弛问题解，nsolution为当前解，obj为当前目标函数值，status为解的状态
	data city;  // 城市信息
	int number;
	float goal;  // 目标解
	float scobj;  // 松弛问题解
	Matrix<float, Dynamic, Dynamic> mat;
	std::vector<int> nsolution;// 当前解，这应该是个n+1维的向量,一般情况下第一位为0，如0-2-3-1-0，表示从0号城市到2号再到3号到1号最后回到0号；
	float obj;  // 当前目标函数值
	std::string status;  // 解的状态
public://四个初始化函数
	TSP(data a, float mubiao = max);
	TSP(Matrix<float, Dynamic, Dynamic> mat, float mubiao = max);
	TSP(Matrix<float, Dynamic, Dynamic> mat, std::vector<int> solution, float mubiao = max);
	TSP(data a, std::vector<int> solution, float mubiao = max);
	float calu() const;
	float showobj() const { return obj; }
	std::vector<int> showsol() const { return nsolution; }
	int shownum() const { return number; }
	//接下来为解的生成方法，相关参数还未输入
	//前4个为初始解的生成方法（非必要），分别为贪心，改进贪心，LKH算法，环生成算法，后三者为精确求解，分别为分支定界、动态规划、回溯算法,再后三个为启发式算法
	//分别为蚁群算法,退火算法,遗传算法，最后一个为强化学习算法（最新研究成果）；
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
		std::cerr << "不是邻接矩阵" << std::endl;
		exit(1);
	}
	number = gmat.rows();
	mat = gmat;
	goal = mubiao;
	obj = max;
}
TSP::TSP(Matrix<float, Dynamic, Dynamic> gmat, std::vector<int> solution, float mubiao) {
	if (mat.rows() != mat.cols()) {
		std::cerr << "不是邻接矩阵" << std::endl;
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
class MTSP {//多旅行商母类，由不同约束条件生成不同子类
private:
	data city;
	float scobj;
	Matrix<int, 1, Dynamic> nsolution;
	float obj;
	bool status;
public:
	MTSP();
};
class MTSP_n :MTSP {//访问数量限制的
private:
	int maxn;
};
//等等,包括其他约束条件