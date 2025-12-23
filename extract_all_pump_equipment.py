from data_preprocessor import CBMDataPreprocessor
import pandas as pd

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== 設置年月日データから全ポンプ設備を抽出 ===')

if processor.installation_dates is not None:
    print(f'設置年月日データ: {len(processor.installation_dates)}台')
    
    # 設置年月日データがある全設備の設備IDリスト
    equipment_with_age_ids = processor.installation_dates['設備ID'].tolist()
    
    # 設備諸元データから対応する設備を検索
    all_equipment_specs = processor.equipment_specs[
        processor.equipment_specs['設備id'].isin(equipment_with_age_ids)
    ]
    
    print(f'設置年月日データがある設備の諸元データ: {len(all_equipment_specs)}行')
    
    # ポンプ関連キーワードで検索
    pump_keywords = ['ポンプ', 'PUMP', 'P-', 'pump', 'Pump']
    pump_equipment_specs = []
    
    for keyword in pump_keywords:
        # 大文字小文字を区別しない検索
        keyword_matches = all_equipment_specs[
            all_equipment_specs['設備名'].str.contains(keyword, case=False, na=False)
        ]
        pump_equipment_specs.append(keyword_matches)
        if len(keyword_matches) > 0:
            unique_equipment = keyword_matches.groupby('設備id')['設備名'].first()
            print(f'キーワード "{keyword}" で見つかった設備: {len(unique_equipment)}台')
            for equip_id, name in unique_equipment.head(5).items():
                print(f'  {equip_id}: {name}')
    
    # 全ポンプ関連設備をまとめる
    all_pump_specs = pd.concat(pump_equipment_specs, ignore_index=True).drop_duplicates()
    
    if len(all_pump_specs) > 0:
        # 設備ごとにグループ化して集計
        pump_equipment_summary = all_pump_specs.groupby(['設備id', '設備名', '設備分類']).agg({
            '測定項目id': 'count',
            '測定回数': 'sum'
        }).reset_index()
        pump_equipment_summary.columns = ['設備id', '設備名', '設備分類', '測定項目数', '総測定回数']
        
        # 設置年月日データと結合
        pump_with_age = pump_equipment_summary.merge(
            processor.installation_dates[['設備ID', '設備年月日']],
            left_on='設備id',
            right_on='設備ID',
            how='left'
        )
        
        # 現在の設備年数を計算
        current_time = pd.Timestamp.now()
        pump_with_age['現在年数'] = (
            (current_time - pump_with_age['設備年月日']).dt.days / 365.25
        )
        
        # 測定回数でソート
        pump_with_age = pump_with_age.sort_values('総測定回数', ascending=False)
        
        print(f'\n=== ポンプ設備抽出結果 ===')
        print(f'設置年月日データがあるポンプ設備: {len(pump_with_age)}台')
        
        print('\n📊 ポンプ設備一覧（測定回数順）:')
        for i, (_, row) in enumerate(pump_with_age.iterrows(), 1):
            print(f'{i:2d}. ID:{row["設備id"]} | {row["設備名"]:<40} | {row["設備分類"]} | {row["現在年数"]:.1f}年 | {row["総測定回数"]:,}回')
        
        # Markdownファイル生成
        md_content = f"""# Pump Equipment List (全データ抽出)
設置年月日データがあるポンプ設備一覧（総計: {len(pump_with_age)}台）

生成日時: 2025年12月23日

## 📊 抽出概要

- **抽出方法**: 設置年月日データがある全設備から、名前にポンプ関連キーワードを含む設備を抽出
- **検索キーワード**: ポンプ, PUMP, P-, pump, Pump
- **総設備数**: {len(pump_with_age)}台
- **年数範囲**: {pump_with_age['現在年数'].min():.1f}年 ～ {pump_with_age['現在年数'].max():.1f}年
- **平均年数**: {pump_with_age['現在年数'].mean():.1f}年

## 📊 設備分類別集計

"""
        
        # 分類別集計
        class_counts = pump_with_age['設備分類'].value_counts()
        for class_name, count in class_counts.items():
            md_content += f"- **{class_name}**: {count}台\n"
        
        md_content += f"""

## 📋 ポンプ設備一覧（測定回数順）

| No. | 設備ID | 設備名 | 設備分類 | 経過年数 | 測定項目数 | 総測定回数 | 設置年月日 |
|-----|--------|--------|----------|----------|------------|------------|------------|
"""
        
        for i, (_, row) in enumerate(pump_with_age.iterrows(), 1):
            installation_str = row['設備年月日'].strftime('%Y-%m-%d') if pd.notna(row['設備年月日']) else 'N/A'
            md_content += f"| {i:2d} | {row['設備id']} | {row['設備名']} | {row['設備分類']} | {row['現在年数']:.1f}年 | {row['測定項目数']}項目 | {row['総測定回数']:,}回 | {installation_str} |\n"
        
        # 詳細情報セクション
        md_content += f"""

## 🔧 設備詳細情報

"""
        
        for i, (_, row) in enumerate(pump_with_age.iterrows(), 1):
            md_content += f"### {i}. {row['設備名']} (ID: {row['設備id']})\n"
            md_content += f"- **設備分類**: {row['設備分類']}\n"
            md_content += f"- **経過年数**: {row['現在年数']:.1f}年\n"
            md_content += f"- **測定項目数**: {row['測定項目数']}項目\n"
            md_content += f"- **総測定回数**: {row['総測定回数']:,}回\n"
            md_content += f"- **設置年月日**: {row['設備年月日'].strftime('%Y年%m月%d日')}\n"
            
            # 測定項目詳細
            measurements = processor.get_measurement_items(row['設備id'])
            if len(measurements) > 0:
                md_content += f"- **測定項目**:\n"
                for _, meas in measurements.iterrows():
                    md_content += f"  - {meas['測定指標']} (ID:{meas['測定項目id']}, {meas['測定回数']:,}回)\n"
            md_content += "\n"
        
        # 推奨テスト設備
        top_5 = pump_with_age.head(5)
        md_content += f"""

## 🎯 推奨テスト設備（測定回数上位5台）

"""
        
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            md_content += f"### {i}. {row['設備名']}\n"
            md_content += f"- **理由**: 測定回数 {row['総測定回数']:,}回で豊富なデータあり\n"
            md_content += f"- **年数**: {row['現在年数']:.1f}年（老朽化考慮可能）\n"
            md_content += f"- **測定項目**: {row['測定項目数']}項目で多角的分析可能\n\n"
        
        # テストコマンド例
        md_content += f"""

## 🎯 CBM強化学習テストコマンド例

```bash
"""
        
        for _, row in pump_with_age.iterrows():
            measurements = processor.get_measurement_items(row['設備id'])
            if len(measurements) > 0:
                main_measurement_id = measurements.iloc[0]['測定項目id']
                # 年数に応じた老朽化係数を計算
                aging_factor = round(max(0.005, 0.008 + (row['現在年数'] - 10) * 0.001), 3)
                md_content += f"# {row['設備名']} ({row['現在年数']:.1f}年)\n"
                md_content += f"python train_cbm_dqn_v2.py --equipment_id {row['設備id']} --measurement_id {main_measurement_id} --episodes 1000 --scenario balanced --aging_factor {aging_factor}\n\n"
        
        md_content += f"""```

## 📈 年数分布分析

"""
        
        # 年数別分布
        age_ranges = [
            (0, 10, "新しい設備"),
            (10, 15, "中程度設備"),
            (15, 20, "経年設備"),
            (20, 25, "老朽化設備"),
            (25, float('inf'), "高齢設備")
        ]
        
        for min_age, max_age, label in age_ranges:
            if max_age == float('inf'):
                count = len(pump_with_age[pump_with_age['現在年数'] >= min_age])
                range_str = f"{min_age}年以上"
            else:
                count = len(pump_with_age[
                    (pump_with_age['現在年数'] >= min_age) & 
                    (pump_with_age['現在年数'] < max_age)
                ])
                range_str = f"{min_age}-{max_age}年"
            
            md_content += f"- **{label}** ({range_str}): {count}台\n"
        
        md_content += f"""

---
*このリストは 全データから抽出したポンプ設備のCBM強化学習テスト用に生成されました。*
*設置年月日データを基に老朽化を考慮した保全戦略学習に活用してください。*
"""
        
        # ファイル保存
        with open('Pump_Equipment_Complete_List.md', 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f'\n✅ Pump_Equipment_Complete_List.md を生成しました')
        print(f'📁 ファイルサイズ: {len(md_content):,} 文字')
        print(f'📊 収録設備数: {len(pump_with_age)}台')
        
    else:
        print('ポンプ関連設備が見つかりませんでした')
        
else:
    print('設置年月日データが読み込まれていません')