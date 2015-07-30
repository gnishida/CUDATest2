/**
 * CUDAの勉強用。
 *
 * 注意：remote desktop 経由だと、結果がおかしくなることがあるようだ。
 *       また、CUDA debuggerも使えない。
 *
 * Projectの右クリックメニューで、Build Customizationsを選択し、CUDA 7.0を選択する
 * ことで、cuファイルに対して自動的にカスタムビルドが設定される！
 */

#include <stdio.h>
#include "vector_test.cuh"
 
int main() {
	cudaMain();

	return 0;
}