"""
5～12年の空調設備候補を特定するスクリプト
"""
import pandas as pd
import os
from datetime import datetime

def find_mid_age_hvac():
    # データファイルのパス確認
    data_dir = 'data/private_benchmark'
    print('=== データファイル確認 ===')
    for file in os.listdir(data_dir):
        print(f'  {file}')

    # 設備設置年データを読み込み
    install_df = pd.read_csv(os.path.join(data_dir, '設備の設置年月日.csv'))
    print(f'設置年データ: {len(install_df)}行')

    # 設備年数を計算
    current_date = datetime(2025, 12, 23)
    install_df['age_years'] = (current_date - pd.to_datetime(install_df['設備年月日'])).dt.days / 365.25

    # 5～12年の設備をフィルタリング
    target_equipment = install_df[(install_df['age_years'] >= 5) & (install_df['age_years'] <= 12)]
    print(f'5-12年設備: {len(target_equipment)}台')

    # 設備諸元データを読み込み
    specs_files = [f for f in os.listdir(data_dir) if '設備諸元' in f]
    if specs_files:
        specs_df = pd.read_csv(os.path.join(data_dir, specs_files[0]))
        print(f'設備諸元データ: {len(specs_df)}行')
        
        # 結合（列名を合わせる）
        merged_df = target_equipment.merge(specs_df[['設備id', '設備名', '測定項目id']], 
                                         left_on='設備ID', right_on='設備id', how='inner')
        
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
        
        print(f'\n=== 5～12年の空調設備候補（上位10台） ===')
        for i, (_, row) in enumerate(hvac_equipment.head(10).iterrows()):
            print(f'{i+1}. 設備ID: {int(row["設備ID"])}, 年数: {row["age_years"]:.1f}年')
            print(f'   設備名: {row["設備名称"]}')
            print(f'   測定項目: {row["測定項目名称"]}')
            print()
            
        # 測定項目IDも取得
        measurements_files = [f for f in os.listdir(data_dir) if '測定値examples' in f]
        if measurements_files:
            measurements_df = pd.read_csv(os.path.join(data_dir, measurements_files[0]))
            
            print("=== 測定項目ID付き候補（推奨2台） ===")
            count = 0
            for _, row in hvac_equipment.iterrows():
                equipment_id = int(row["設備ID"])
                # 測定項目IDを検索
                measurement_matches = measurements_df[measurements_df['設備ID'] == equipment_id]
                if not measurement_matches.empty:
                    measurement_id = measurement_matches['測定項目ID'].iloc[0]
                    print(f'設備ID: {equipment_id}, 測定項目ID: {measurement_id}, 年数: {row["age_years"]:.1f}年')
                    print(f'設備名: {row["設備名称"]}')
                    print(f'測定項目: {row["測定項目名称"]}')
                    print()
                    
                    count += 1
                    if count >= 2:
                        break

if __name__ == "__main__":
    find_mid_age_hvac()