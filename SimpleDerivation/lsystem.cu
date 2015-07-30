#include <stdio.h>
#include <curand_kernel.h>
#include "lsystem.cuh"
#include "vector.cuh"

#define LENGTH_ATTENUATION					0.9

__host__ __device__
Literal::Literal(const string& name, int depth, double param_value1, double param_value2, double param_value3, double param_value4) {
	this->name = name;
	this->depth = depth;
	this->param_values.push_back(param_value1);
	this->param_values.push_back(param_value2);
	this->param_values.push_back(param_value3);
	this->param_values.push_back(param_value4);
	this->param_defined = true;
}

__host__ __device__
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

__host__ __device__
Literal::Literal(const string& name, int depth, const vector<double>& param_values) {
	this->name = name;
	this->depth = depth;
	this->param_values = param_values;
	this->param_defined = true;
}

__host__ __device__
String Literal::operator+(const Literal& l) const {
	String ret = *this;
	return ret + l;
}

__host__ __device__
int Literal::type() {
	if (name == "F" || name == "f" || name == "C" || name == "[" || name == "]" || name == "+" || name == "-" || name == "\\" || name == "/" || name == "&" || name == "^" || name == "#") {
		return TYPE_TERMINAL;
	} else {
		return TYPE_NONTERMINAL;
	}
}

__host__ __device__
String::String() {
	this->cursor = -1;
}

__host__ __device__
String::String(const string& str, int depth) {
	this->str.push_back(Literal(str, depth));
	this->cursor = 0;
}

__host__ __device__
String::String(const Literal& l) {
	this->str.push_back(l);
	this->cursor = 0;
}

__host__ __device__
void String::operator+=(const Literal& l) {
	str.push_back(l);

	if (cursor < 0) cursor = 0;
}

__host__ __device__
void String::operator+=(const String& str) {
	for (int i = 0; i < str.length(); ++i) {
		this->str.push_back(str[i]);
	}

	if (cursor < 0) cursor = 0;
}

__host__ __device__
String String::operator+(const String& str) const {
	String new_str = *this;

	for (int i = 0; i < str.length(); ++i) {
		new_str.str.push_back(str[i]);
	}

	if (new_str.cursor < 0 && new_str.length() > 0) new_str.cursor = 0;

	return new_str;
}

__host__ __device__
void String::setValue(double value) {
	str[cursor].param_values.push_back(value);
	str[cursor].param_defined = true;

	cursor++;

	// 次のリテラルを探す
	nextCursor(str[cursor].depth);
}

__host__ __device__
void String::replace(const String& str) {
	int depth = this->str[cursor].depth;

	this->str.erase(this->str.begin() + cursor);
	this->str.insert(this->str.begin() + cursor, str.str.begin(), str.str.end());
	this->str.insert(this->str.begin() + cursor, this->str.begin() + cursor, this->str.begin() + cursor);

	// 次のリテラルを探す
	nextCursor(depth);
}

__host__ __device__
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

__host__ __device__
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

__host__ __device__
Action::Action(int action_index, int index, const String& rule) {
	this->type = ACTION_RULE;
	this->action_index = action_index;
	this->index = index;
	this->rule = rule;
}

__host__ __device__
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
__host__ __device__
String Action::apply(const String& model) {
	String new_model = model;

	if (type == ACTION_RULE) {
		new_model.replace(rule);
	} else {
		new_model.setValue(value);
	}

	return new_model;
}

__host__ __device__
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

