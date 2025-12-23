import sys
from data_preprocessor import CBMDataPreprocessor

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== 老朽化データありの設備を検索（ポンプ関連） ===')
available_equipment = processor.get_available_equipment_with_age()

# ポンプ関連設備を検索
pump_equipment = []
for equipment_id, data in available_equipment.items():
    equipment_name = data['equipment_name']
    if 'ポンプ' in equipment_name or 'PUMP' in equipment_name.upper() or 'P-' in equipment_name:
        pump_equipment.append({
            'equipment_id': equipment_id,
            'equipment_name': equipment_name,
            'measurements': data['measurements'],
            'age_years': data['age_years']
        })

print(f'見つかったポンプ設備: {len(pump_equipment)}台')
print()

for i, equip in enumerate(pump_equipment[:10]):
    print(f'{i+1}. 設備ID: {equip["equipment_id"]}')
    print(f'   設備名: {equip["equipment_name"]}')
    print(f'   経過年数: {equip["age_years"]:.1f}年')
    print(f'   測定項目数: {len(equip["measurements"])}項目')
    
    measurement_list = list(equip['measurements'].items())
    for j, (meas_id, meas_name) in enumerate(measurement_list[:3]):
        print(f'     - {meas_name} (ID: {meas_id})')
    if len(equip['measurements']) > 3:
        print(f'     ... 他{len(equip["measurements"])-3}項目')
    print()