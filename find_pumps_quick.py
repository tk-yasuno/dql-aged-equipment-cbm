import sys
from data_preprocessor import CBMDataPreprocessor

# より短いテスト
processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('データロード完了')

# まず全設備の数を確認（空調設備で検索）
available_equipment_df = processor.get_available_equipment_with_age("空調設備")
print(f'空調設備で老朽化データありの設備総数: {len(available_equipment_df)}台')

if len(available_equipment_df) == 0:
    print('老朽化データがある設備が見つかりません')
    # 別のクラスも試してみる
    for equipment_class in ['空調設備', '電気設備', 'その他']:
        df = processor.get_available_equipment_with_age(equipment_class)
        print(f'{equipment_class}: {len(df)}台')
else:
    # ポンプ関連設備を検索
    pump_patterns = ['ポンプ', 'PUMP', 'P-', 'pump']
    found_pumps = []
    
    for _, row in available_equipment_df.iterrows():
        equipment_name = row['設備名']
        for pattern in pump_patterns:
            if pattern in equipment_name or pattern.upper() in equipment_name.upper():
                found_pumps.append((row['設備id'], equipment_name, row['現在年数']))
                break
    
    print(f'\nポンプ関連設備: {len(found_pumps)}台')
    for equipment_id, name, age in found_pumps[:5]:
        print(f'  {equipment_id}: {name} ({age:.1f}年)')