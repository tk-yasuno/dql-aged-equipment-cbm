from data_preprocessor import CBMDataPreprocessor
import pandas as pd

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== Lifetime Equipment List 生成中 ===')

# 空調設備のデータ（既に確認済み）を使用
available_equipment_df = processor.get_available_equipment_with_age("空調設備")
print(f'空調設備: {len(available_equipment_df)}台')

# 年数順でソート
available_equipment_df = available_equipment_df.sort_values('現在年数', ascending=False)

# Markdownファイルを生成
md_content = f"""# Lifetime Equipment List
設置年月日データがある設備一覧（空調設備: {len(available_equipment_df)}台）

生成日時: 2025年12月23日

## 📊 設備概要

- **設備分類**: 空調設備のみ
- **総設備数**: {len(available_equipment_df)}台
- **年数範囲**: {available_equipment_df['現在年数'].min():.1f}年 ～ {available_equipment_df['現在年数'].max():.1f}年
- **平均年数**: {available_equipment_df['現在年数'].mean():.1f}年

## 📋 全設備一覧（年数順）

| No. | 設備ID | 設備名 | 経過年数 | 測定項目数 | 総測定回数 | 設置年月日 |
|-----|--------|--------|----------|------------|------------|------------|
"""

# 各設備の詳細を追加
for i, (_, row) in enumerate(available_equipment_df.iterrows(), 1):
    installation_str = row['設備年月日'].strftime('%Y-%m-%d') if pd.notna(row['設備年月日']) else 'N/A'
    md_content += f"| {i:3d} | {row['設備id']} | {row['設備名']} | {row['現在年数']:.1f}年 | {row['測定項目数']}項目 | {row['総測定回数']:,}回 | {installation_str} |\n"

# 測定回数上位設備の推奨リスト
top_10 = available_equipment_df.head(10)

md_content += f"""

## 📈 推奨テスト対象設備（測定回数上位10台）

"""

for i, (_, row) in enumerate(top_10.iterrows(), 1):
    md_content += f"### {i}. {row['設備名']} (ID: {row['設備id']})\n"
    md_content += f"- **経過年数**: {row['現在年数']:.1f}年\n"
    md_content += f"- **測定項目数**: {row['測定項目数']}項目\n"
    md_content += f"- **総測定回数**: {row['総測定回数']:,}回\n"
    
    # 主要測定項目を取得（簡略版）
    measurements = processor.get_measurement_items(row['設備id'])
    if len(measurements) > 0:
        top_measurement = measurements.iloc[0]
        md_content += f"- **主要測定項目**: {top_measurement['測定指標']} (ID:{top_measurement['測定項目id']})\n"
    md_content += "\n"

# 年数別分布
age_ranges = [
    (0, 10, "新しい設備"),
    (10, 15, "中程度設備"),
    (15, 18, "経年設備"),
    (18, 20, "老朽化設備"),
    (20, float('inf'), "高齢設備")
]

md_content += f"""

## 📊 年数別分布

"""

for min_age, max_age, label in age_ranges:
    if max_age == float('inf'):
        count = len(available_equipment_df[available_equipment_df['現在年数'] >= min_age])
        range_str = f"{min_age}年以上"
    else:
        count = len(available_equipment_df[
            (available_equipment_df['現在年数'] >= min_age) & 
            (available_equipment_df['現在年数'] < max_age)
        ])
        range_str = f"{min_age}-{max_age}年"
    
    md_content += f"- **{label}** ({range_str}): {count}台\n"

# テストコマンド例
top_5 = available_equipment_df.head(5)
md_content += f"""

## 🎯 CBM強化学習テストコマンド例

以下は測定回数上位設備でのテストコマンド例です：

```bash
"""

for _, row in top_5.iterrows():
    measurements = processor.get_measurement_items(row['設備id'])
    if len(measurements) > 0:
        main_measurement_id = measurements.iloc[0]['測定項目id']
        aging_factor = 0.015 if row['現在年数'] > 18 else 0.012
        md_content += f"# {row['設備名']} ({row['現在年数']:.1f}年)\n"
        md_content += f"python train_cbm_dqn_v2.py --equipment_id {row['設備id']} --measurement_id {main_measurement_id} --episodes 1000 --scenario balanced --aging_factor {aging_factor}\n\n"

md_content += f"""```

## 📝 設備タイプ別特徴

### 冷却器系設備 (R-series)
- **R-1-1 ～ R-2-2**: 冷水入口/出口温度、凝縮器圧力を監視
- **特徴**: 高い測定頻度、温度・圧力の複合監視
- **推奨用途**: 温度制御システムのCBM学習

### エアハンドリングユニット (AHU-series) 
- **AHU-TSK-A/B/C**: プレフィルタ差圧、電流、温度を監視
- **特徴**: 差圧監視によるフィルタ状態判定
- **推奨用途**: フィルタ保全戦略の学習

### 外気処理機 (OAC-series)
- **OAC-TSK-F**: 外気処理専用、フィルタ・温度制御
- **特徴**: 外気条件の影響を受けやすい
- **推奨用途**: 環境変動を考慮したCBM学習

---
*このリストは CBM強化学習システムの設備選定資料として生成されました。*
*測定データの品質と頻度を考慮して、適切な設備を選択してください。*
"""

# Markdownファイルを保存
with open('Lifetime_equipment_List.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f'✅ Lifetime_equipment_List.md を生成しました')
print(f'📁 ファイルサイズ: {len(md_content):,} 文字')
print(f'📊 収録設備数: {len(available_equipment_df)}台')