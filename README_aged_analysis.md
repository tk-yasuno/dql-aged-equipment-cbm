# Equipment CBM with Aging Analysis
設備老朽化を考慮したCBM強化学習システム

作成日時: 2025年12月23日

## 📋 プロジェクト概要

このフォルダーには、設備の老朽化リスクを考慮したCondition-Based Maintenance（CBM）強化学習システムの改良版が含まれています。

### 🎯 改良の目的
- **問題点**: 従来版では設備の経過年数データが含まれておらず、老朽化が考慮されていなかった
- **改善点**: 設備設置年月日データを統合し、老朽化リスクを考慮した保全戦略学習を実現
- **対象設備**: 設置年月日データがある205台の設備（空調設備202台 + ポンプ設備3台）

## 🔄 改良の経緯

### Phase 1: データ統合 (12/23)
1. **設置年月日データ追加**
   - `data/private_benchmark/設備の設置年月日.csv` を統合
   - 測定日時における経過年数を動的計算
   - 設備ID突合による年数データ付与

2. **データ前処理拡張** (`data_preprocessor.py`)
   - `get_equipment_age()`: 設備年数計算機能
   - `get_available_equipment_with_age()`: 老朽化データがある設備一覧取得
   - `estimate_age_adjusted_parameters()`: 年数考慮パラメータ推定

### Phase 2: 環境拡張 (12/23)
3. **3D状態空間への拡張** (`cbm_environment.py`)
   - 従来: 2D (condition, temperature)
   - 改良: 3D (condition, temperature, normalized_age)
   - 老朽化による状態遷移確率の動的調整

4. **老朽化効果モデリング**
   - `_get_age_adjusted_transition()`: 年数依存遷移確率
   - 設備交換時の年数リセット機能
   - Aging factorによる劣化速度制御

### Phase 3: 学習システム強化 (12/23)
5. **学習アルゴリズム対応** (`train_cbm_dqn_v2.py`)
   - 3D観測空間対応
   - 老朽化パラメータ設定機能
   - 年数情報を含む学習履歴記録

6. **可視化システム拡張** (`visualize_results.py`)
   - 老朽化分析プロット追加
   - 3D状態空間対応
   - 年数-異常率相関分析

### Phase 4: 対象設備選定 (12/23)
7. **設備カテゴリ分析**
   - 機械設備: 0台（設置年データなし）→対象外
   - 空調設備: 202台（データあり）→主要対象
   - ポンプ設備: 3台（空調分類内）→特別対象

8. **設備リスト生成**
   - `Lifetime_equipment_List.md`: 空調設備202台リスト
   - `Lifetime_pump_List.md`: ポンプ設備3台リスト

## 📊 対象設備データ

### 空調設備 (202台)
- **冷却器系**: R-1-1 ～ R-2-2 (19.7年、高測定頻度)
- **AHU系**: AHU-TSK-A/B/C (15-16年、差圧・温度監視)
- **外気処理機**: OAC-TSK-F (17年、フィルタ管理)

### ポンプ設備 (3台)
| 設備名 | 年数 | 測定回数 | 特徴 |
|--------|------|----------|------|
| 薬注ポンプCP-500-5 | 19.7年 | 1,029回 | 老朽化設備 |
| 冷却水ポンプCDP-A5 | 3.0年 | 1,002回 | 新しい設備 |
| 薬注ポンプCP-500-3 | 0.5年 | 157回 | 最新設備 |

## 🔧 改良されたファイル構成

### コアシステム
- `data_preprocessor.py`: 老朽化データ統合機能付きデータ前処理
- `cbm_environment.py`: 3D状態空間対応強化学習環境
- `train_cbm_dqn_v2.py`: 老朽化考慮学習スクリプト
- `visualize_results.py`: 老朽化分析可視化システム

### 設定・データ
- `config.yaml`: 老朽化パラメータ設定
- `outputs_cbm_v2/`: 2000エピソード学習結果

### ドキュメント
- `Lifetime_equipment_List.md`: 空調設備完全リスト
- `Lifetime_pump_List.md`: ポンプ設備詳細リスト

### 分析用スクリプト
- `list_hvac_equipment.py`: 空調設備一覧生成
- `extract_all_pump_equipment.py`: ポンプ設備抽出
- `check_mechanical_equipment.py`: 機械設備データ確認

## 🎯 実行例

### 老朽化設備でのテスト
```bash
# R-1-1冷却器 (19.7年) - 2000エピソード実行済み
python train_cbm_dqn_v2.py --equipment_id 265693 --measurement_id 258863 --episodes 2000 --scenario balanced --aging_factor 0.015

# 結果可視化
python visualize_results.py --output_dir outputs_cbm_v2 --equipment_id 265693 --measurement_id 258863
```

### ポンプ設備比較テスト
```bash
# 老朽化ポンプ (19.7年)
python train_cbm_dqn_v2.py --equipment_id 265715 --measurement_id [ID] --episodes 1500 --scenario balanced --aging_factor 0.018

# 新しいポンプ (3.0年)  
python train_cbm_dqn_v2.py --equipment_id 137953 --measurement_id [ID] --episodes 1500 --scenario balanced --aging_factor 0.005
```

## 📈 学習成果

### 実証された効果
- **老朽化リスクの定量化**: 年数と異常率の正の相関を確認
- **3D状態空間の有効性**: 温度・状態・年数の統合分析
- **動的保全戦略**: 設備年数に応じた最適行動学習

### 可視化結果 (R-1-1, 2000エピソード)
- 平均報酬: -37.40
- 状態分布: Normal 40.6%, Anomalous 59.4%
- 年数-異常率相関: 0.126 (正の相関確認)

## 🚀 今後の展開

### 短期目標
1. **多設備並行学習**: 複数設備での比較実験
2. **最適化戦略分析**: 年数別保全戦略の差異分析
3. **コスト効果測定**: 老朽化考慮による保全コスト削減効果

### 長期ビジョン
1. **予測保全システム**: リアルタイム老朽化リスク評価
2. **設備ライフサイクル管理**: 全設備の統合保全戦略
3. **AIによる設備投資判断**: 更新・修理の最適タイミング予測

## 📚 参考資料

- **従来版**: `../equipment-cbm-mvp/` (老朽化考慮なし)
- **ベース研究**: `../base_markov-dqn-v09-quantile/` (QR-DQN実装)
- **データソース**: `../data/private_benchmark/` (設備諸元・測定値・設置年月日)

---
*設備老朽化を考慮したCBM強化学習により、予防保全戦略の革新的進歩を目指しています。*
