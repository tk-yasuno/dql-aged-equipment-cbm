from data_preprocessor import CBMDataPreprocessor

processor = CBMDataPreprocessor('../data/private_benchmark')
processor.load_data()

# 対象設備リスト
target_equipment = [
    (265715, "薬注ポンプCP-500-5", 19.7, "ポンプ設備"),
    (137953, "冷却水ポンプCDP-A5", 3.0, "ポンプ設備"), 
    (519177, "薬注ポンプCP-500-3", 0.5, "ポンプ設備"),
    (327240, "AHU-TSK-A-2", 15.6, "空調設備"),
    (265694, "R-1-3", 19.7, "空調設備"),
    (322220, "OAC-TSK-F-2", 17.7, "空調設備")
]

print("=== 6台の対象設備の測定項目確認 ===")

for equipment_id, name, age, category in target_equipment:
    print(f"\n📊 {name} (ID: {equipment_id}, {age}年)")
    print(f"   カテゴリ: {category}")
    
    measurements = processor.get_measurement_items(equipment_id)
    if len(measurements) > 0:
        print(f"   測定項目数: {len(measurements)}項目")
        print("   主要測定項目:")
        for i, (_, row) in enumerate(measurements.head(3).iterrows(), 1):
            print(f"     {i}. {row['測定指標']} (ID: {row['測定項目id']}, {row['測定回数']:,}回)")
        
        # 推奨測定項目（最も測定回数が多い項目）
        top_measurement = measurements.iloc[0]
        
        # 年数に応じたaging_factor計算
        if age < 1:
            aging_factor = 0.003
        elif age < 5:
            aging_factor = 0.005
        elif age < 15:
            aging_factor = 0.010
        elif age < 18:
            aging_factor = 0.015
        else:
            aging_factor = 0.018
            
        print(f"   ✅ 推奨: 測定項目ID {top_measurement['測定項目id']} (aging_factor: {aging_factor})")
        
        # 実行コマンド生成
        print(f"   🚀 実行コマンド:")
        print(f"   python train_cbm_dqn_v2.py --equipment_id {equipment_id} --measurement_id {top_measurement['測定項目id']} --episodes 1000 --scenario balanced --aging_factor {aging_factor}")
        
    else:
        print(f"   ⚠️ 測定項目が見つかりません")

print(f"\n=== 実行準備完了 ===")
print("上記のコマンドを順次実行してください")