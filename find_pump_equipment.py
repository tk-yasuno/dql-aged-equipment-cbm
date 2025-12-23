from data_preprocessor import CBMDataPreprocessor
import pandas as pd

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== ポンプ設備の設置年データ調査 ===')

# 全設備分類で老朽化データがある設備を調査
equipment_classes = ['機械設備', '空調設備', '電気設備', 'その他']
all_pump_equipment = []

for equipment_class in equipment_classes:
    df = processor.get_available_equipment_with_age(equipment_class)
    if len(df) > 0:
        print(f'\n{equipment_class}: {len(df)}台中からポンプ設備を検索')
        
        # ポンプ関連キーワードで検索
        pump_keywords = ['ポンプ', 'PUMP', 'P-', 'pump', '19']
        
        for _, row in df.iterrows():
            equipment_name = row['設備名']
            equipment_id = str(row['設備id'])
            
            # ポンプキーワードまたは19idで始まる設備を検索
            is_pump = False
            matched_keyword = None
            
            for keyword in pump_keywords:
                if (keyword in equipment_name or 
                    keyword.upper() in equipment_name.upper() or
                    (keyword == '19' and equipment_id.startswith('19'))):
                    is_pump = True
                    matched_keyword = keyword
                    break
            
            if is_pump:
                all_pump_equipment.append({
                    'class': equipment_class,
                    'equipment_id': row['設備id'],
                    'equipment_name': row['設備名'],
                    'age_years': row['現在年数'],
                    'measurement_items': row['測定項目数'],
                    'total_measurements': row['総測定回数'],
                    'installation_date': row['設備年月日'],
                    'matched_keyword': matched_keyword
                })
                print(f'  見つけた: {row["設備id"]} - {equipment_name} ({matched_keyword}で一致)')

print(f'\n=== 検索結果 ===')
print(f'ポンプ関連設備: {len(all_pump_equipment)}台')

if len(all_pump_equipment) == 0:
    print('\nポンプ設備が見つかりませんでした。19idで始まる設備を詳細調査します...')
    
    # 19で始まる設備IDを詳細調査
    for equipment_class in equipment_classes:
        df = processor.get_available_equipment_with_age(equipment_class)
        if len(df) > 0:
            id19_equipment = df[df['設備id'].astype(str).str.startswith('19')]
            if len(id19_equipment) > 0:
                print(f'\n{equipment_class}で19idの設備: {len(id19_equipment)}台')
                for _, row in id19_equipment.iterrows():
                    print(f'  {row["設備id"]}: {row["設備名"]} ({row["現在年数"]:.1f}年)')

# ポンプ設備が見つかった場合、リストを生成
if len(all_pump_equipment) > 0:
    # 年数順でソート
    all_pump_equipment.sort(key=lambda x: x['age_years'], reverse=True)
    
    # Markdownファイル生成
    md_content = f"""# Pump Equipment List
ポンプ設備の設置年月日データ一覧（総計: {len(all_pump_equipment)}台）

生成日時: 2025年12月23日

## 📊 設備概要

- **検索対象**: ポンプ関連設備（名前にポンプ/PUMP/P-含む、または19idで始まる設備）
- **総設備数**: {len(all_pump_equipment)}台

### 設備分類別集計

"""

    # 分類別集計
    class_counts = {}
    for equip in all_pump_equipment:
        class_name = equip['class']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

    for class_name, count in class_counts.items():
        md_content += f"- **{class_name}**: {count}台\n"

    # 年数統計
    ages = [e['age_years'] for e in all_pump_equipment]
    md_content += f"""

### 年数統計
- **最古設備**: {max(ages):.1f}年
- **最新設備**: {min(ages):.1f}年  
- **平均年数**: {sum(ages)/len(ages):.1f}年

## 📋 ポンプ設備一覧（年数順）

| No. | 設備分類 | 設備ID | 設備名 | 経過年数 | 測定項目数 | 総測定回数 | マッチ理由 |
|-----|----------|--------|--------|----------|------------|------------|------------|
"""

    # 各設備の詳細
    for i, equip in enumerate(all_pump_equipment, 1):
        md_content += f"| {i:2d} | {equip['class']} | {equip['equipment_id']} | {equip['equipment_name']} | {equip['age_years']:.1f}年 | {equip['measurement_items']}項目 | {equip['total_measurements']:,}回 | {equip['matched_keyword']} |\n"

    # 詳細情報
    md_content += f"""

## 🔧 設備詳細情報

"""

    for i, equip in enumerate(all_pump_equipment, 1):
        md_content += f"### {i}. {equip['equipment_name']} (ID: {equip['equipment_id']})\n"
        md_content += f"- **設備分類**: {equip['class']}\n"
        md_content += f"- **経過年数**: {equip['age_years']:.1f}年\n"
        md_content += f"- **測定項目数**: {equip['measurement_items']}項目\n"
        md_content += f"- **総測定回数**: {equip['total_measurements']:,}回\n"
        
        # 測定項目詳細
        measurements = processor.get_measurement_items(equip['equipment_id'])
        if len(measurements) > 0:
            md_content += f"- **測定項目**:\n"
            for _, meas in measurements.iterrows():
                md_content += f"  - {meas['測定指標']} (ID:{meas['測定項目id']}, {meas['測定回数']:,}回)\n"
        md_content += "\n"

    # テストコマンド例
    md_content += f"""

## 🎯 CBM強化学習テストコマンド例

```bash
"""

    for equip in all_pump_equipment:
        measurements = processor.get_measurement_items(equip['equipment_id'])
        if len(measurements) > 0:
            main_measurement_id = measurements.iloc[0]['測定項目id']
            aging_factor = 0.015 if equip['age_years'] > 15 else 0.010
            md_content += f"# {equip['equipment_name']} ({equip['age_years']:.1f}年)\n"
            md_content += f"python train_cbm_dqn_v2.py --equipment_id {equip['equipment_id']} --measurement_id {main_measurement_id} --episodes 1000 --scenario balanced --aging_factor {aging_factor}\n\n"

    md_content += """```

---
*このリストは ポンプ設備のCBM強化学習テスト用に生成されました。*
"""

    # ファイル保存
    with open('Pump_Equipment_List.md', 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f'\n✅ Pump_Equipment_List.md を生成しました')
    print(f'📁 ファイルサイズ: {len(md_content):,} 文字')
    
else:
    print('ポンプ設備の設置年データが見つかりませんでした。')