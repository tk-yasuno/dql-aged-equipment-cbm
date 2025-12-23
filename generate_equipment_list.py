import sys
from data_preprocessor import CBMDataPreprocessor
import pandas as pd

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== 設置年月日データがある全設備の調査 ===')

# 各設備分類での老朽化データありの設備を確認
equipment_classes = ['機械設備', '空調設備', '電気設備', 'その他']
all_equipment = []

for equipment_class in equipment_classes:
    df = processor.get_available_equipment_with_age(equipment_class)
    if len(df) > 0:
        print(f'{equipment_class}: {len(df)}台')
        for _, row in df.iterrows():
            all_equipment.append({
                'class': equipment_class,
                'equipment_id': row['設備id'],
                'equipment_name': row['設備名'],
                'age_years': row['現在年数'],
                'measurement_items': row['測定項目数'],
                'total_measurements': row['総測定回数'],
                'installation_date': row['設備年月日']
            })
    else:
        print(f'{equipment_class}: 0台')

print(f'\n総計: {len(all_equipment)}台の設備で老朽化データ有り')

# 年数順でソート
all_equipment.sort(key=lambda x: x['age_years'], reverse=True)

# Markdownファイルを生成
md_content = f"""# Lifetime Equipment List
設置年月日データがある設備一覧（総計: {len(all_equipment)}台）

生成日時: 2025年12月23日

## 📊 設備分類別集計

"""

# 分類別集計を追加
class_counts = {}
for equip in all_equipment:
    class_name = equip['class']
    if class_name not in class_counts:
        class_counts[class_name] = 0
    class_counts[class_name] += 1

for class_name, count in class_counts.items():
    md_content += f"- **{class_name}**: {count}台\n"

md_content += f"""

## 📋 全設備一覧（年数順）

| No. | 設備分類 | 設備ID | 設備名 | 経過年数 | 測定項目数 | 総測定回数 | 設置年月日 |
|-----|----------|--------|--------|----------|------------|------------|------------|
"""

# 各設備の詳細を追加
for i, equip in enumerate(all_equipment, 1):
    installation_str = equip['installation_date'].strftime('%Y-%m-%d') if pd.notna(equip['installation_date']) else 'N/A'
    md_content += f"| {i:3d} | {equip['class']} | {equip['equipment_id']} | {equip['equipment_name']} | {equip['age_years']:.1f}年 | {equip['measurement_items']}項目 | {equip['total_measurements']:,}回 | {installation_str} |\n"

# 設備分類別の詳細セクションを追加
for equipment_class in equipment_classes:
    class_equipment = [e for e in all_equipment if e['class'] == equipment_class]
    if len(class_equipment) > 0:
        md_content += f"""

## 🔧 {equipment_class} 詳細 ({len(class_equipment)}台)

"""
        for i, equip in enumerate(class_equipment, 1):
            md_content += f"### {i}. {equip['equipment_name']} (ID: {equip['equipment_id']})\n"
            md_content += f"- **経過年数**: {equip['age_years']:.1f}年\n"
            md_content += f"- **測定項目数**: {equip['measurement_items']}項目\n"
            md_content += f"- **総測定回数**: {equip['total_measurements']:,}回\n"
            
            # 測定項目の詳細を取得
            measurements = processor.get_measurement_items(equip['equipment_id'])
            if len(measurements) > 0:
                md_content += f"- **主要測定項目**:\n"
                top_measurements = measurements.head(5)  # 上位5項目
                for _, meas in top_measurements.iterrows():
                    md_content += f"  - {meas['測定指標']} (ID:{meas['測定項目id']}, {meas['測定回数']:,}回)\n"
            md_content += "\n"

md_content += f"""

## 📈 推奨テスト対象設備

### 高頻度測定設備（上位10台）
測定回数が多く、学習に適した設備：

"""

# 測定回数上位10台を推奨として追加
top_10 = sorted(all_equipment, key=lambda x: x['total_measurements'], reverse=True)[:10]
for i, equip in enumerate(top_10, 1):
    md_content += f"{i:2d}. **{equip['equipment_name']}** (ID: {equip['equipment_id']})\n"
    md_content += f"    - 分類: {equip['class']}\n"
    md_content += f"    - 年数: {equip['age_years']:.1f}年\n"
    md_content += f"    - 測定回数: {equip['total_measurements']:,}回\n\n"

md_content += f"""

### 年数別分布
老朽化の影響を比較するための年数分布：

"""

# 年数別の分布を追加
age_ranges = [
    (0, 5, "新しい設備"),
    (5, 10, "中程度の経過"),
    (10, 15, "経年設備"),
    (15, 20, "老朽化設備"),
    (20, float('inf'), "高齢設備")
]

for min_age, max_age, label in age_ranges:
    if max_age == float('inf'):
        count = len([e for e in all_equipment if e['age_years'] >= min_age])
        range_str = f"{min_age}年以上"
    else:
        count = len([e for e in all_equipment if min_age <= e['age_years'] < max_age])
        range_str = f"{min_age}-{max_age}年"
    
    md_content += f"- **{label}** ({range_str}): {count}台\n"

md_content += f"""

## 🎯 CBM強化学習テストコマンド例

以下は主要設備でのテストコマンド例です：

```bash
# R-1-1 (冷却器、19.7年)
python train_cbm_dqn_v2.py --equipment_id 265693 --measurement_id 258863 --episodes 1000 --scenario balanced --aging_factor 0.015

# AHU-TSK-A-2 (エアハンドリングユニット、15.6年)
python train_cbm_dqn_v2.py --equipment_id 327240 --measurement_id 353609 --episodes 1000 --scenario balanced --aging_factor 0.012

# R-1-3 (冷却器、19.7年)
python train_cbm_dqn_v2.py --equipment_id 265694 --measurement_id 258887 --episodes 1000 --scenario balanced --aging_factor 0.015

# AHU-TSK-B-1 (エアハンドリングユニット、15.4年)
python train_cbm_dqn_v2.py --equipment_id 327241 --measurement_id 353630 --episodes 1000 --scenario balanced --aging_factor 0.012
```

---
*このリストは CBM強化学習システムのテスト対象設備選定のために生成されました。*
"""

# Markdownファイルを保存
with open('Lifetime_equipment_List.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print(f'\n✅ Lifetime_equipment_List.md を生成しました')
print(f'📁 ファイルサイズ: {len(md_content):,} 文字')
print(f'📊 収録設備数: {len(all_equipment)}台')