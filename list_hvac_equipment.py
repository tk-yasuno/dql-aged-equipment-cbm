import sys
from data_preprocessor import CBMDataPreprocessor

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

print('=== 空調設備で老朽化データありの設備一覧 ===')
available_equipment_df = processor.get_available_equipment_with_age("空調設備")
print(f'空調設備総数: {len(available_equipment_df)}台')

# 測定回数の多い上位15台を表示
top_equipment = available_equipment_df.head(15)

print('\n📊 測定回数上位15台の空調設備:')
print('-' * 80)
for i, (_, row) in enumerate(top_equipment.iterrows()):
    print(f'{i+1:2d}. 設備ID: {row["設備id"]:6d} | {row["設備名"]:<40} | {row["現在年数"]:.1f}年')
    print(f'     測定項目: {row["測定項目数"]:2d}項目 | 総測定回数: {row["総測定回数"]:,}回')
    
    # この設備の測定項目を確認
    measurements = processor.get_measurement_items(row["設備id"])
    if len(measurements) > 0:
        # 上位3つの測定項目を表示
        top_measurements = measurements.head(3)
        for _, meas in top_measurements.iterrows():
            print(f'       - {meas["測定指標"]} (ID:{meas["測定項目id"]}, {meas["測定回数"]:,}回)')
    print()