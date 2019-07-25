# Research of Hyperparameter Optimization

M2研究用レポジトリ．SOMET2019の研究を基にメタヒューリスティックアルゴリズムを用いてCNNのHyperparameter最適化を行う．

- test/ 実験用レポジトリ
- src/  公開時に用いるレポジトリ

## Use data

SOMET2019の研究で用いた4-class methodのtimewindow1.5sの画像を用いる．同じ画像枚数で比較できるようにするためには同じtimewindowである必要があり，各timewindowのAccracyの最も低いものが最もなるものを用いた．

- male: method 4-class, timewindow 1.5s
- female: method 4-class, timewindow 1.5s

### memo (2019/07/25)

- 今回は2019/06/04論文を基にする
- 最適化の例として宇宙テザーのやつも論文に入れたい
- SOMETの内容を第一段階の実験とし、それを基にMethod2の画像を用いてPSO最適化実験をしてまとめる
- どのような提案PSO?
    - ノーマルPSOは既に例があるので（2019/06/04論文），Wight decay PSOを使う?
- 結果は何と比較?
    - PSOを使わなかった自分のSOMET論文の結果と比較
- 実装はHyperactiveを編集して作成