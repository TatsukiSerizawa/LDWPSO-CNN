# Research of Hyperparameter Optimization

M2研究用レポジトリ．SOMET2019の研究を基にメタヒューリスティックアルゴリズムを用いてCNNのHyperparameter最適化を行う．

- test/ 実験用レポジトリ
- src/  公開時に用いるレポジトリ

## Use data

SOMET2019の研究で用いた4-class methodのtimewindow1.5sの画像を用いる．同じ画像枚数で比較できるようにするためには同じtimewindowである必要があり，各timewindowのAccracyの最も低いものが最もなるものを用いた．

- male: method 4-class, timewindow 1.5s
- female: method 4-class, timewindow 1.5s