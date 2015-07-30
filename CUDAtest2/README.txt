CUDAを使用するProjectを新規作成する方法：
1. 普通にProjectを作成する。
2. Projectの右クリックメニューから[Build Customizations...]を選択する。
3. [Cuda 7.0]を選択し、OKをクリックする。
4. [Source Files]の右クリックメニューから[Add] -> [New Item...]を選択する。
5. 画面左のリストから、[NVIDA CUDA 7.0] -> [Code]を選択し、画面中央のリストから、[CUDA C/C++ File]を選択し、ファイル名を入力して、新規CUファイルを作成する。
6. ProjectのProperty画面で、[Linker] -> [Input]を選択し、[Additional Dependencies]で[cudart.lib]を入力する。