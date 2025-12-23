"""
AHU系設備から5～12年の候補2台を選定
"""

import pandas as pd
import os
from datetime import datetime

# データ読み込み
data_dir = 'data/private_benchmark'
install_df = pd.read_csv(os.path.join(data_dir, '設備の設置年月日.csv'))
specs_df = pd.read_csv(os.path.join(data_dir, '設備諸元_実測値100以上.csv'))

print("AHU系設備の候補選定を開始...")

# AHU系設備をフィルタリング
ahu_mask = install_df['設備名称'].str.contains('AHU|エアハン', case=False, na=False)
ahu_equipment = install_df[ahu_mask].copy()
print(f'AHU系設備: {len(ahu_equipment)}台')

# 測定項目が存在する設備のみに絞る
ahu_with_measurement = ahu_equipment.merge(
    specs_df[['設備id', '測定項目id']],
    left_on='設備ID',
    right_on='設備id',
    how='inner'
)
print(f'測定項目が存在するAHU設備: {len(ahu_with_measurement)}台')

# 年数計算
current_date = datetime(2025, 12, 23)
ahu_with_measurement['設備年月日_clean'] = pd.to_datetime(ahu_with_measurement['設備年月日'], errors='coerce')
valid_ahu = ahu_with_measurement.dropna(subset=['設備年月日_clean'])
valid_ahu = valid_ahu[valid_ahu['設備年月日_clean'] >= '1990-01-01']
valid_ahu['age_years'] = (current_date - valid_ahu['設備年月日_clean']).dt.days / 365.25

print(f'有効日付のAHU設備: {len(valid_ahu)}台')

if len(valid_ahu) > 0:
    print(f'年数範囲: {valid_ahu["age_years"].min():.1f}～{valid_ahu["age_years"].max():.1f}年')
    
    # 5～15年の範囲で候補を探す（12年だと少なすぎる可能性があるので範囲拡大）
    target_ahu = valid_ahu[(valid_ahu['age_years'] >= 5) & (valid_ahu['age_years'] <= 15)]
    print(f'5-15年AHU設備: {len(target_ahu)}台')
    
    if len(target_ahu) >= 2:
        target_sorted = target_ahu.sort_values('age_years')
        
        print('\n=== 候補AHU設備 TOP5 ===')
        for i, (_, row) in enumerate(target_sorted.head(5).iterrows()):
            equipment_id = int(row['設備ID'])
            measurement_id = int(row['測定項目id'])
            age_years = row['age_years']
            equipment_name = row['設備名称']
            
            print(f'{i+1}. 設備ID: {equipment_id}, 測定項目ID: {measurement_id}')
            print(f'   年数: {age_years:.1f}年, 設備名: {equipment_name}')
            print()
        
        # 最終推奨2台
        print('=== 最終推奨2台 ===')
        candidate1 = target_sorted.iloc[0]  # 最も若い
        candidate2 = target_sorted.iloc[len(target_sorted)//2] if len(target_sorted) > 2 else target_sorted.iloc[1]  # 中間
        
        for i, candidate in enumerate([candidate1, candidate2], 1):
            equipment_id = int(candidate['設備ID'])
            measurement_id = int(candidate['測定項目id'])
            age_years = candidate['age_years']
            equipment_name = candidate['設備名称']
            
            # AHU系のaging_factor推奨値
            aging_factor = 0.008 + age_years * 0.0005
            
            print(f'推奨{i}: 設備ID {equipment_id}, 測定項目ID {measurement_id}')
            print(f'       年数: {age_years:.1f}年')
            print(f'       aging_factor: {aging_factor:.4f}')
            print(f'       設備名: {equipment_name}')
            print(f'       出力ディレクトリ: outputs_ahu_{equipment_id}')
            print()
    else:
        print(f'5-15年の範囲にAHU設備が{len(target_ahu)}台しかありません')
        if len(valid_ahu) >= 2:
            print('範囲を拡大して候補を表示します:')
            for i, (_, row) in enumerate(valid_ahu.head(5).iterrows()):
                equipment_id = int(row['設備ID'])
                measurement_id = int(row['測定項目id'])
                age_years = row['age_years']
                equipment_name = row['設備名称']
                
                print(f'{i+1}. 設備ID: {equipment_id}, 測定項目ID: {measurement_id}')
                print(f'   年数: {age_years:.1f}年, 設備名: {equipment_name}')
else:
    print('有効なAHU設備データがありません')