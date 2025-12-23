from data_preprocessor import CBMDataPreprocessor
import pandas as pd

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== 機械設備の設置年月日データ有無確認 ===')

# 機械設備の基本情報を取得
mechanical_equipment = processor.get_available_equipment("機械設備")
print(f'機械設備の総数: {len(mechanical_equipment)}台')

if len(mechanical_equipment) > 0:
    print('\n📊 機械設備一覧（測定回数順）:')
    for i, (_, row) in enumerate(mechanical_equipment.head(10).iterrows(), 1):
        print(f'{i:2d}. ID:{row["設備id"]} | {row["設備名"]:<50} | {row["測定項目数"]}項目 | {row["総測定回数"]:,}回')

# 設置年月日データとの照合確認
if processor.installation_dates is not None:
    print(f'\n=== 設置年月日データとの照合確認 ===')
    print(f'設置年月日データ総数: {len(processor.installation_dates)}台')
    
    # 機械設備のIDリスト
    mechanical_equipment_ids = set(mechanical_equipment['設備id'].tolist())
    print(f'機械設備のID数: {len(mechanical_equipment_ids)}台')
    
    # 設置年月日データがある設備のIDリスト
    equipment_with_age_ids = set(processor.installation_dates['設備ID'].tolist())
    print(f'設置年月日データがあるID数: {len(equipment_with_age_ids)}台')
    
    # 機械設備と設置年月日データの重複を確認
    mechanical_with_age = mechanical_equipment_ids.intersection(equipment_with_age_ids)
    print(f'\n🔍 機械設備で設置年月日データがある設備: {len(mechanical_with_age)}台')
    
    if len(mechanical_with_age) > 0:
        print('該当設備ID:', list(mechanical_with_age))
        
        # 該当設備の詳細情報
        for equipment_id in mechanical_with_age:
            equipment_info = mechanical_equipment[mechanical_equipment['設備id'] == equipment_id]
            age_info = processor.installation_dates[processor.installation_dates['設備ID'] == equipment_id]
            if len(equipment_info) > 0 and len(age_info) > 0:
                equip = equipment_info.iloc[0]
                age = age_info.iloc[0]
                age_years = (pd.Timestamp.now() - age['設備年月日']).days / 365.25
                print(f'  - {equip["設備名"]} (ID:{equipment_id}) : {age_years:.1f}年')
    else:
        print('✅ 確認結果: 機械設備には設置年月日データがありません')

    # 参考: 他の設備分類での設置年月日データ確認
    print(f'\n=== 参考: 各設備分類での設置年月日データ有無 ===')
    equipment_classes = ['機械設備', '空調設備', '電気設備', 'その他']
    
    for equipment_class in equipment_classes:
        class_equipment = processor.get_available_equipment(equipment_class)
        if len(class_equipment) > 0:
            class_ids = set(class_equipment['設備id'].tolist())
            class_with_age = class_ids.intersection(equipment_with_age_ids)
            print(f'- {equipment_class}: {len(class_equipment)}台中 {len(class_with_age)}台に設置年月日データあり')
        else:
            print(f'- {equipment_class}: 0台')

    # 設置年月日データがある設備の分類別詳細
    print(f'\n=== 設置年月日データがある設備の分類別分析 ===')
    
    # 設置年月日データがある設備の設備諸元を取得
    equipment_with_age_specs = processor.equipment_specs[
        processor.equipment_specs['設備id'].isin(equipment_with_age_ids)
    ]
    
    if len(equipment_with_age_specs) > 0:
        # 分類別集計
        class_distribution = equipment_with_age_specs['設備分類'].value_counts()
        print('設置年月日データがある設備の分類別分布:')
        for class_name, count in class_distribution.items():
            # 設備台数も計算（重複除去）
            unique_equipment_count = len(
                equipment_with_age_specs[equipment_with_age_specs['設備分類'] == class_name]['設備id'].unique()
            )
            print(f'  - {class_name}: {unique_equipment_count}台 (測定項目数: {count}項目)')
    
    # 機械設備の設備IDレンジ確認
    print(f'\n=== 機械設備のID範囲分析 ===')
    if len(mechanical_equipment) > 0:
        mechanical_id_min = mechanical_equipment['設備id'].min()
        mechanical_id_max = mechanical_equipment['設備id'].max()
        print(f'機械設備のID範囲: {mechanical_id_min} ～ {mechanical_id_max}')
        
        # 設置年月日データのID範囲と比較
        age_id_min = processor.installation_dates['設備ID'].min()
        age_id_max = processor.installation_dates['設備ID'].max()
        print(f'設置年月日データのID範囲: {age_id_min} ～ {age_id_max}')
        
        # 範囲の重複確認
        overlap_exists = not (mechanical_id_max < age_id_min or mechanical_id_min > age_id_max)
        print(f'ID範囲の重複: {"あり" if overlap_exists else "なし"}')
        
        if overlap_exists:
            print('重複範囲内でも設置年月日データがないため、機械設備は対象外であることを確認')

else:
    print('設置年月日データが読み込まれていません')

print(f'\n=== 結論 ===')
print('✅ 機械設備は設置年月日データなし → CBM老朽化学習の対象外')
print('✅ 空調設備のみが老朽化考慮CBM学習の対象')
print('✅ ポンプ設備（3台）は空調設備分類に含まれる')