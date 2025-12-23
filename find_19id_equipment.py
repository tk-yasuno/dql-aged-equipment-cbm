from data_preprocessor import CBMDataPreprocessor
import pandas as pd

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== 19idで始まる設備の調査 ===')

# 設備諸元データから19で始まる設備IDを直接検索
equipment_specs = processor.equipment_specs
print(f'全設備諸元データ: {len(equipment_specs)}行')

# 19で始まる設備IDをフィルタ
id19_specs = equipment_specs[equipment_specs['設備id'].astype(str).str.startswith('19')]
print(f'19idで始まる設備: {len(id19_specs)}行（測定項目含む）')

if len(id19_specs) > 0:
    # 設備ごとにグループ化
    id19_equipment = id19_specs.groupby(['設備id', '設備名']).agg({
        '測定項目id': 'count',
        '測定回数': 'sum'
    }).reset_index()
    id19_equipment.columns = ['設備id', '設備名', '測定項目数', '総測定回数']
    id19_equipment = id19_equipment.sort_values('総測定回数', ascending=False)
    
    print(f'19idの設備数: {len(id19_equipment)}台')
    print(f'設備ID範囲: {id19_equipment["設備id"].min()} - {id19_equipment["設備id"].max()}')
    
    print('\n=== 19id設備一覧（測定回数順） ===')
    for _, row in id19_equipment.head(10).iterrows():
        print(f'{row["設備id"]}: {row["設備名"]} ({row["測定項目数"]}項目, {row["総測定回数"]:,}回)')

# 設置年月日データとの照合
if processor.installation_dates is not None:
    print(f'\n=== 設置年月日データとの照合 ===')
    print(f'設置年月日データ: {len(processor.installation_dates)}行')
    
    # 19idの設備で設置年月日データがあるものを検索
    id19_with_age = processor.installation_dates[
        processor.installation_dates['設備ID'].astype(str).str.startswith('19')
    ]
    print(f'19idで設置年データあり: {len(id19_with_age)}台')
    
    if len(id19_with_age) > 0:
        print('\n設置年データがある19id設備:')
        for _, row in id19_with_age.head(10).iterrows():
            age = (pd.Timestamp.now() - row['設備年月日']).days / 365.25
            print(f'  {row["設備ID"]}: {age:.1f}年 (設置: {row["設備年月日"].strftime("%Y-%m-%d")})')
        
        # 設備諸元と結合してポンプ設備リストを作成
        pump_equipment = []
        for _, age_row in id19_with_age.iterrows():
            equipment_id = age_row['設備ID']
            equipment_info = id19_equipment[id19_equipment['設備id'] == equipment_id]
            
            if len(equipment_info) > 0:
                info = equipment_info.iloc[0]
                age_years = (pd.Timestamp.now() - age_row['設備年月日']).days / 365.25
                
                pump_equipment.append({
                    'equipment_id': equipment_id,
                    'equipment_name': info['設備名'],
                    'age_years': age_years,
                    'measurement_items': info['測定項目数'],
                    'total_measurements': info['総測定回数'],
                    'installation_date': age_row['設備年月日']
                })
        
        # 年数順でソート
        pump_equipment.sort(key=lambda x: x['age_years'], reverse=True)
        
        print(f'\n=== ポンプ設備リスト生成 ===')
        print(f'対象設備: {len(pump_equipment)}台')
        
        if len(pump_equipment) > 0:
            # Markdownファイル生成
            md_content = f"""# Pump Equipment List (19id系設備)
19idで始まる設備の設置年月日データ一覧（総計: {len(pump_equipment)}台）

生成日時: 2025年12月23日

## 📊 設備概要

- **対象**: 19で始まる設備ID
- **総設備数**: {len(pump_equipment)}台
- **年数範囲**: {min(e['age_years'] for e in pump_equipment):.1f}年 ～ {max(e['age_years'] for e in pump_equipment):.1f}年
- **平均年数**: {sum(e['age_years'] for e in pump_equipment)/len(pump_equipment):.1f}年

## 📋 設備一覧（年数順）

| No. | 設備ID | 設備名 | 経過年数 | 測定項目数 | 総測定回数 | 設置年月日 |
|-----|--------|--------|----------|------------|------------|------------|
"""

            for i, equip in enumerate(pump_equipment, 1):
                installation_str = equip['installation_date'].strftime('%Y-%m-%d')
                md_content += f"| {i:2d} | {equip['equipment_id']} | {equip['equipment_name']} | {equip['age_years']:.1f}年 | {equip['measurement_items']}項目 | {equip['total_measurements']:,}回 | {installation_str} |\n"

            # 詳細情報
            md_content += f"""

## 🔧 設備詳細情報

"""

            for i, equip in enumerate(pump_equipment, 1):
                md_content += f"### {i}. {equip['equipment_name']} (ID: {equip['equipment_id']})\n"
                md_content += f"- **経過年数**: {equip['age_years']:.1f}年\n"
                md_content += f"- **測定項目数**: {equip['measurement_items']}項目\n"
                md_content += f"- **総測定回数**: {equip['total_measurements']:,}回\n"
                md_content += f"- **設置年月日**: {equip['installation_date'].strftime('%Y年%m月%d日')}\n"
                
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

            for equip in pump_equipment:
                measurements = processor.get_measurement_items(equip['equipment_id'])
                if len(measurements) > 0:
                    main_measurement_id = measurements.iloc[0]['測定項目id']
                    aging_factor = round(0.010 + (equip['age_years'] - 10) * 0.001, 3)  # 年数に応じた老朽化係数
                    md_content += f"# {equip['equipment_name']} ({equip['age_years']:.1f}年)\n"
                    md_content += f"python train_cbm_dqn_v2.py --equipment_id {equip['equipment_id']} --measurement_id {main_measurement_id} --episodes 1000 --scenario balanced --aging_factor {aging_factor}\n\n"

            md_content += """```

---
*このリストは 19id系ポンプ設備のCBM強化学習テスト用に生成されました。*
"""

            # ファイル保存
            with open('Pump_Equipment_19id_List.md', 'w', encoding='utf-8') as f:
                f.write(md_content)

            print(f'✅ Pump_Equipment_19id_List.md を生成しました')
            print(f'📁 ファイルサイズ: {len(md_content):,} 文字')
            print(f'📊 収録設備数: {len(pump_equipment)}台')
        else:
            print('19id設備で測定データが十分な設備が見つかりませんでした')
    else:
        print('19id設備で設置年データがある設備が見つかりませんでした')
else:
    print('設置年月日データが読み込まれていません')