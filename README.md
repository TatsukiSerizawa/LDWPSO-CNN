# Research of Hyperparameter Optimization

M2研究用レポジトリ．SOMET2019の研究を基にメタヒューリスティックアルゴリズムを用いてCNNのHyperparameter最適化を行う．

- test/ 実験用レポジトリ
- src/  公開時に用いるレポジトリ

## Use data

SOMET2019の研究で用いた4-class methodのtimewindow1.5sの画像を用いる．同じ画像枚数で比較できるようにするためには同じtimewindowである必要があり，各timewindowのAccracyの最も低いものが最もなるものを用いた．

- male: method 4-class, timewindow 1.5s
- female: method 4-class, timewindow 1.5s

## memo (2019/07/25)

- 今回は2019/06/04論文を基にする
- 最適化の例として宇宙テザーのやつも論文に入れたい
- SOMETの内容を第一段階の実験とし、それを基にMethod2の画像を用いてPSO最適化実験をしてまとめる
- どのような提案PSO?
    - ノーマルPSOは既に例があるので（2019/06/04論文），Wight decay PSOを使う?  
      →コスト重視でRandom methodを使うか，一般性重視でLinear decay methodを使うか
- 結果は何と比較?
    - PSOを使わなかった自分のSOMET論文の結果と比較
- 実装はHyperactiveを編集して作成

## memo (2019/07/30)

PSO in previous research
- Paper: Convolutional neural network-based PSO for lung nodule false positive reduction on CT images (201906/04)
- Optimaization hyperparameter
    - Number of filters in Conv. 1, 2 (4~100)
    - Number of neurons in the hidden layer (4~100)
    - Size of kernel in Conv. 1, 2 (3, 5, 7)
    - Type of pooling in Pooling1, 2 (max-pooling, average-pooling)
    - Batch size in the training (10~100)
    - Probability of dropout in convolutional layer (0.1~0.99)
    - Probability of dropout in completely connected layer (0.1~0.99)
- Parameters selected for PSO algorithm
    - Swam size: 10
    - Number of iterations: 30
    - Cognitive parametr: 2.0
    - Social parameter: 2.0
    - Inertia weight: 0.7
- Training process
    - Optimizer: SGD
    - Larning rate: 0.01

