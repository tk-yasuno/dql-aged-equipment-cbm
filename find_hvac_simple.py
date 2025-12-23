"""
5～12年の空調設備候補を特定するシンプルスクリプト
"""

import pandas as pd
import os
from datetime import datetime

# データファイルのパス
data_dir = 'data/private_benchmark'

# 設備設置年データを読み込み
install_df = pd.read_csv(os.path.join(data_dir, '設備の設置年月日.csv'))
print(f'設置年データ: {len(install_df)}行')

# 設備年数を計算（異常な日付を除外）
current_date = datetime(2025, 12, 23)

# 日付データを先に変換し、異常値を除外
install_df['設備年月日_clean'] = pd.to_datetime(install_df['設備年月日'], errors='coerce')

# 現実的な日付範囲でフィルタリング（1990年以降）
valid_date_mask = (install_df['設備年月日_clean'] >= '1990-01-01') & \
                  (install_df['設備年月日_clean'] <= current_date)
install_df_clean = install_df[valid_date_mask].copy()
print(f'有効日付データ: {len(install_df_clean)}行')

# 年数計算
install_df_clean['age_years'] = (current_date - install_df_clean['設備年月日_clean']).dt.days / 365.25

# 5～12年の設備をフィルタリング
target_equipment = install_df_clean[(install_df_clean['age_years'] >= 5) & (install_df_clean['age_years'] <= 12)]
print(f'5-12年設備: {len(target_equipment)}台')

# 設備諸元データを読み込み
specs_df = pd.read_csv(os.path.join(data_dir, '設備諸元_実測値100以上.csv'))
print(f'設備諸元データ: {len(specs_df)}行')

# 結合（列名を合わせる）
merged_df = target_equipment.merge(specs_df[['設備id', '設備名', '測定項目id']], 
                                 left_on='設備ID', right_on='設備id', how='inner')
print(f'結合後データ: {len(merged_df)}行')

# 空調設備キーワード
hvac_keywords = ['AHU', 'PAU', 'FAU', 'OAC', 'エアハンドリング', '外気処理', '給気', '排気', 'ユニット']
hvac_mask = merged_df['設備名'].str.contains('|'.join(hvac_keywords), case=False, na=False)
hvac_equipment = merged_df[hvac_mask].sort_values('age_years')
print(f'5～12年空調設備: {len(hvac_equipment)}台')

print(f'\n=== 5～12年の空調設備候補 TOP10 ===')
for i, (_, row) in enumerate(hvac_equipment.head(10).iterrows()):
    equipment_id = int(row['設備ID'])
    measurement_id = int(row['測定項目id'])
    age_years = row['age_years']
    equipment_name = row['設備名']
    
    print(f'{i+1:2d}. 設備ID: {equipment_id:6d}, 測定項目ID: {measurement_id:6d}, 年数: {age_years:4.1f}年')
    print(f'     設備名: {equipment_name}')
    print()
    
# 推奨2台（年数の違いを考慮）
print('=== 推奨候補2台 ===')
if len(hvac_equipment) >= 2:
    # 年数範囲を分けて選択
    young_hvac = hvac_equipment[hvac_equipment['age_years'] <= 8]  # 5-8年
    old_hvac = hvac_equipment[hvac_equipment['age_years'] > 8]     # 8-12年
    
    candidates = []
    if len(young_hvac) > 0:
        candidates.append(young_hvac.iloc[0])
    if len(old_hvac) > 0:
        candidates.append(old_hvac.iloc[0])
    elif len(young_hvac) > 1:
        candidates.append(young_hvac.iloc[1])
    
    for i, candidate in enumerate(candidates, 1):
        equipment_id = int(candidate['設備ID'])
        measurement_id = int(candidate['測定項目id'])
        age_years = candidate['age_years']
        equipment_name = candidate['設備名']
        
        print(f'推奨{i}: 設備ID {equipment_id}, 測定項目ID {measurement_id}, 年数 {age_years:.1f}年')
        print(f'       設備名: {equipment_name}')
        print()