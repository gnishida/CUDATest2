#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include "vector.cuh"
#include "string.cuh"

#define LENGTH_ATTENUATION		0.9

class String;

class Literal {
public:
	static enum { TYPE_TERMINAL = 0, TYPE_NONTERMINAL};

public:
	string name;
	int depth;
	vector<double> param_values;
	bool param_defined;

public:
	CUDA_CALLABLE_MEMBER
	Literal() {}

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, bool param_defined = false);

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, double param_value);

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, double param_value1, double param_value2);

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3);

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3, double param_value4);

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3, double param_value4, double param_value5);

	CUDA_CALLABLE_MEMBER
	Literal(const string& name, int depth, const vector<double>& param_values);

	CUDA_CALLABLE_MEMBER
	String operator+(const Literal& l) const;

	CUDA_CALLABLE_MEMBER
	int type();
};

class String {
public:
	vector<Literal> str;
	int cursor;				// expandするリテラルの位置

public:
	CUDA_CALLABLE_MEMBER
	String();

	CUDA_CALLABLE_MEMBER
	String(const string& str, int depth);

	CUDA_CALLABLE_MEMBER
	String(const Literal& l);

	CUDA_CALLABLE_MEMBER
	String& operator=(const Literal& l);

	CUDA_CALLABLE_MEMBER
	int length() const { return str.size(); }

	CUDA_CALLABLE_MEMBER
	Literal operator[](int index) const { return str[index]; }

	CUDA_CALLABLE_MEMBER
	Literal& operator[](int index) { return str[index]; }

	CUDA_CALLABLE_MEMBER
	void operator+=(const Literal& l);

	CUDA_CALLABLE_MEMBER
	void operator+=(const String& str);

	CUDA_CALLABLE_MEMBER
	String operator+(const String& str) const;

	CUDA_CALLABLE_MEMBER
	void setValue(double value);

	CUDA_CALLABLE_MEMBER
	void replace(const String& str);

	CUDA_CALLABLE_MEMBER
	String getExpand() const;

	CUDA_CALLABLE_MEMBER
	void nextCursor(int depth);
};

/**
 * 1ステップのderiveを表す。
 */
class Action {
public:
	static enum { ACTION_RULE = 0, ACTION_VALUE };

public:
	int type;		// 0 -- rule / 1 -- value
	int index;		// モデルの何文字目の変数に対するactionか？
	int action_index;	// actionsの中の何番目のactionか？
	String rule;
	double value;

public:
	CUDA_CALLABLE_MEMBER
	Action() {}

	CUDA_CALLABLE_MEMBER
	Action(int action_index, int index, const String& rule);

	CUDA_CALLABLE_MEMBER
	Action(int action_index, int index, double value);

	CUDA_CALLABLE_MEMBER
	String apply(const String& model);
};

CUDA_CALLABLE_MEMBER
vector<Action> getActions(const String& model);









////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, bool param_defined) {
	this->name = name;
	this->depth = depth;
	this->param_defined = param_defined;
}

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, double param_value) {
	this->name = name;
	this->depth = depth;
	this->param_values.push_back(param_value);
	this->param_defined = true;
}

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, double param_value1, double param_value2) {
	this->name = name;
	this->depth = depth;
	this->param_values.push_back(param_value1);
	this->param_values.push_back(param_value2);
	this->param_defined = true;
}

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3) {
	this->name = name;
	this->depth = depth;
	this->param_values.push_back(param_value1);
	this->param_values.push_back(param_value2);
	this->param_values.push_back(param_value3);
	this->param_defined = true;
}

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3, double param_value4) {
	this->name = name;
	this->depth = depth;
	this->param_values.push_back(param_value1);
	this->param_values.push_back(param_value2);
	this->param_values.push_back(param_value3);
	this->param_values.push_back(param_value4);
	this->param_defined = true;
}

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3, double param_value4, double param_value5) {
	this->name = name;
	this->depth = depth;
	this->param_values.push_back(param_value1);
	this->param_values.push_back(param_value2);
	this->param_values.push_back(param_value3);
	this->param_values.push_back(param_value4);
	this->param_values.push_back(param_value5);
	this->param_defined = true;
}

CUDA_CALLABLE_MEMBER
Literal::Literal(const string& name, int depth, const vector<double>& param_values) {
	this->name = name;
	this->depth = depth;
	this->param_values = param_values;
	this->param_defined = true;
}

CUDA_CALLABLE_MEMBER
String Literal::operator+(const Literal& l) const {
	String ret = *this;
	return ret + l;
}

CUDA_CALLABLE_MEMBER
int Literal::type() {
	if (name == "F" || name == "f" || name == "C" || name == "[" || name == "]" || name == "+" || name == "-" || name == "\\" || name == "/" || name == "&" || name == "^" || name == "#") {
		return TYPE_TERMINAL;
	} else {
		return TYPE_NONTERMINAL;
	}
}

CUDA_CALLABLE_MEMBER
String::String() {
	this->cursor = -1;
}

CUDA_CALLABLE_MEMBER
String::String(const string& str, int depth) {
	this->str.push_back(Literal(str, depth));
	this->cursor = 0;
}

CUDA_CALLABLE_MEMBER
String::String(const Literal& l) {
	this->str.push_back(l);
	this->cursor = 0;
}

CUDA_CALLABLE_MEMBER
void String::operator+=(const Literal& l) {
	str.push_back(l);

	if (cursor < 0) cursor = 0;
}

CUDA_CALLABLE_MEMBER
void String::operator+=(const String& str) {
	for (int i = 0; i < str.length(); ++i) {
		this->str.push_back(str[i]);
	}

	if (cursor < 0) cursor = 0;
}

CUDA_CALLABLE_MEMBER
String String::operator+(const String& str) const {
	String new_str = *this;

	for (int i = 0; i < str.length(); ++i) {
		new_str.str.push_back(str[i]);
	}

	if (new_str.cursor < 0 && new_str.length() > 0) new_str.cursor = 0;

	return new_str;
}

CUDA_CALLABLE_MEMBER
void String::setValue(double value) {
	str[cursor].param_values.push_back(value);
	str[cursor].param_defined = true;

	cursor++;

	// 次のリテラルを探す
	nextCursor(str[cursor].depth);
}

CUDA_CALLABLE_MEMBER
void String::replace(const String& str) {
	int depth = this->str[cursor].depth;

	this->str.erase(this->str.begin() + cursor);
	this->str.insert(this->str.begin() + cursor, str.str.begin(), str.str.end());
	this->str.insert(this->str.begin() + cursor, this->str.begin() + cursor, this->str.begin() + cursor);

	// 次のリテラルを探す
	nextCursor(depth);
}

CUDA_CALLABLE_MEMBER
String String::getExpand() const {
	String ret;

	int nest = 0;
	for (int i = cursor; i < str.size(); ++i) {
		if (str[i].name == "[") {
			nest++;
		} else if (str[i].name == "]") {
			nest--;
		}

		if (nest < 0) break;

		ret += str[i];
	}

	return ret;
}

CUDA_CALLABLE_MEMBER
void String::nextCursor(int depth) {
	for (int i = cursor; i < str.size(); ++i) {
		if (str[i].depth != depth) continue;

		if (str[i].type() == Literal::TYPE_NONTERMINAL) {
			cursor = i;
			return;
		} else if (str[i].type() == Literal::TYPE_TERMINAL && !str[i].param_defined) {
			cursor = i;
			return;
		}
	}

	// 同じdepthでリテラルが見つからない場合は、depth+1にする
	depth++;

	for (int i = 0; i < str.size(); ++i) {
		if (str[i].depth != depth) continue;

		if (str[i].type() == Literal::TYPE_NONTERMINAL) {
			cursor = i;
			return;
		} else if (str[i].type() == Literal::TYPE_TERMINAL && !str[i].param_defined) {
			cursor = i;
			return;
		}
	}

	// リテラルが見つからない場合は、-1にする
	cursor = -1;
}

CUDA_CALLABLE_MEMBER
Action::Action(int action_index, int index, const String& rule) {
	this->type = ACTION_RULE;
	this->action_index = action_index;
	this->index = index;
	this->rule = rule;
}

CUDA_CALLABLE_MEMBER
Action::Action(int action_index, int index, double value) {
	this->type = ACTION_VALUE;
	this->action_index = action_index;
	this->index = index;
	this->value = value;
}

/**
 * 指定されたモデルに、このアクションを適用する。
 *
 * @param model					モデル
 * @return						action適用した後のモデル
 */
CUDA_CALLABLE_MEMBER
String Action::apply(const String& model) {
	String new_model = model;

	if (type == ACTION_RULE) {
		new_model.replace(rule);
	} else {
		new_model.setValue(value);
	}

	return new_model;
}

CUDA_CALLABLE_MEMBER
vector<Action> getActions(const String& model) {
	vector<Action> actions;

	// 展開するパラメータを決定
	int i = model.cursor;

	// 新たなderivationがないなら、終了
	if (i == -1) return actions;

	if (model[i].name == "X") {
		String rule = Literal("F", model[i].depth + 1, model[i].param_values[0], model[i].param_values[1])
			+ Literal("#", model[i].depth + 1)
			+ Literal("\\", model[i].depth + 1, 50.0)
			+ Literal("X", model[i].depth + 1, model[i].param_values[0] * LENGTH_ATTENUATION, model[i].param_values[1] + model[i].param_values[0]);
		actions.push_back(Action(actions.size(), i, rule));

		rule = Literal("F", model[i].depth + 1, model[i].param_values[0] * 0.5f, model[i].param_values[1])
			+ Literal("[", model[i].depth + 1, true)
			+ Literal("+", model[i].depth + 1)
			+ Literal("X", model[i].depth + 1, model[i].param_values[0] * LENGTH_ATTENUATION, model[i].param_values[1] + model[i].param_values[0] * 0.5f)
			+ Literal("]", model[i].depth + 1, true)
			+ Literal("F", model[i].depth + 1, model[i].param_values[0] * 0.5f, model[i].param_values[1] + model[i].param_values[0] * 0.5f)
			+ Literal("#", model[i].depth + 1)
			+ Literal("\\", model[i].depth + 1, 50.0)
			+ Literal("X", model[i].depth + 1, model[i].param_values[0] * LENGTH_ATTENUATION, model[i].param_values[1] + model[i].param_values[0]);
		actions.push_back(Action(actions.size(), i, rule));
	} else if (model[i].name == "+" || model[i].name == "-") {
		for (int k = -80; k <= 80; k += 20) {
			if (k == 0) continue;
			actions.push_back(Action(actions.size(), i, k));
		}
	}

	return actions;
}

