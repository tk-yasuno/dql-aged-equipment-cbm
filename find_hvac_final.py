"""
5～12年の空調設備を特定し、学習用の設備ID・測定項目IDペアを出力
"""

import pandas as pd
import os
from datetime import datetime

def main():
    data_dir = 'data/private_benchmark'
    
    # 設備設置年データ読み込み
    install_df = pd.read_csv(os.path.join(data_dir, '設備の設置年月日.csv'))
    print(f'設置年データ: {len(install_df)}行')
    
    # 空調設備をフィルタリング
    hvac_keywords = ['AHU', 'PAU', 'FAU', 'OAC', 'エアハンドリング', '外気処理', '給気', '排気', 'ユニット']
    hvac_mask = install_df['設備名称'].str.contains('|'.join(hvac_keywords), case=False, na=False)
    hvac_equipment = install_df[hvac_mask].copy()
    print(f'空調設備候補: {len(hvac_equipment)}台')
    
    # 年数計算（有効日付のみ）
    current_date = datetime(2025, 12, 23)
    hvac_equipment['設備年月日_clean'] = pd.to_datetime(hvac_equipment['設備年月日'], errors='coerce')
    valid_hvac = hvac_equipment.dropna(subset=['設備年月日_clean'])
    valid_hvac = valid_hvac[valid_hvac['設備年月日_clean'] >= '1990-01-01']
    valid_hvac['age_years'] = (current_date - valid_hvac['設備年月日_clean']).dt.days / 365.25
    
    print(f'有効日付の空調設備: {len(valid_hvac)}台')
    
    if len(valid_hvac) > 0:
        print(f'年数範囲: {valid_hvac["age_years"].min():.1f}～{valid_hvac["age_years"].max():.1f}年')
        
        # 年数分布表示
        print('\n年数分布:')
        for start_age in range(0, 35, 5):
            end_age = start_age + 5
            count = len(valid_hvac[(valid_hvac['age_years'] >= start_age) & (valid_hvac['age_years'] < end_age)])
            print(f'{start_age}-{end_age}年: {count}台')
        
        # 5～15年の設備を抽出（範囲を少し広げる）
        target_hvac = valid_hvac[(valid_hvac['age_years'] >= 5) & (valid_hvac['age_years'] <= 15)]
        print(f'\n5-15年空調設備: {len(target_hvac)}台')
        
        if len(target_hvac) > 0:
            # 設備諸元データと結合して測定項目IDを取得
            specs_df = pd.read_csv(os.path.join(data_dir, '設備諸元_実測値100以上.csv'))
            merged_df = target_hvac.merge(
                specs_df[['設備id', '設備名', '測定項目id']],
                left_on='設備ID',
                right_on='設備id',
                how='inner'
            )
            
            print(f'測定項目が存在する5-15年空調設備: {len(merged_df)}台')
            
            if len(merged_df) > 0:
                # 年数順でソート
                merged_sorted = merged_df.sort_values('age_years')
                
                print('\n=== 5-15年空調設備候補（測定項目付き）===')
                for i, (_, row) in enumerate(merged_sorted.head(8).iterrows()):
                    equipment_id = int(row['設備ID'])
                    measurement_id = int(row['測定項目id'])
                    age_years = row['age_years']
                    equipment_name = row['設備名称']
                    
                    print(f'{i+1:2d}. 設備ID: {equipment_id:6d}, 測定項目ID: {measurement_id:6d}, 年数: {age_years:4.1f}年')
                    print(f'     設備名: {equipment_name}')
                    print()
                
                # 推奨2台を選択
                print('=== 推奨2台（年数の異なるもの）===')
                if len(merged_sorted) >= 2:
                    # 若い方から1台、古い方から1台
                    candidate1 = merged_sorted.iloc[0]
                    candidate2 = merged_sorted.iloc[-1]
                    
                    for i, candidate in enumerate([candidate1, candidate2], 1):
                        equipment_id = int(candidate['設備ID'])
                        measurement_id = int(candidate['測定項目id'])
                        age_years = candidate['age_years']
                        equipment_name = candidate['設備名称']
                        aging_factor = 0.008 + age_years * 0.0005  # AHU系の推奨aging_factor
                        
                        print(f'推奨{i}: 設備ID {equipment_id}, 測定項目ID {measurement_id}')
                        print(f'       年数: {age_years:.1f}年, aging_factor推奨: {aging_factor:.4f}')
                        print(f'       設備名: {equipment_name}')
                        print()
            else:
                print('測定項目が存在する設備がありません')
        else:
            print('該当年数の空調設備がありません')
    else:
        print('有効な日付データがありません')

if __name__ == '__main__':
    main()