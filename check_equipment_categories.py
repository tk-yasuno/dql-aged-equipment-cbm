"""
設備分類の確認と年数データがある設備分類の特定
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data_preprocessor import CBMDataPreprocessor

def check_equipment_categories():
    """利用可能な設備分類と年数データの組み合わせを確認"""
    
    data_dir = Path(__file__).parent.parent / "data" / "private_benchmark"
    preprocessor = CBMDataPreprocessor(data_dir)
    preprocessor.load_data()
    
    print("="*80)
    print("📋 設備分類別分析")
    print("="*80)
    
    # 全ての設備分類を取得
    equipment_categories = preprocessor.equipment_specs['設備分類'].unique()
    print(f"利用可能な設備分類 ({len(equipment_categories)}種類):")
    for i, category in enumerate(equipment_categories, 1):
        count = len(preprocessor.equipment_specs[preprocessor.equipment_specs['設備分類'] == category])
        print(f"  {i:2d}. {category:<20} ({count:3d}件)")
    
    print("\n" + "="*60)
    print("📊 各分類での年数データ有無確認")
    print("="*60)
    
    # 各分類で年数データがある設備を確認
    for category in equipment_categories:
        equipment_with_age = preprocessor.get_available_equipment_with_age(category)
        equipment_count = len(preprocessor.get_available_equipment(category))
        age_count = len(equipment_with_age)
        
        status = "✅" if age_count > 0 else "❌"
        print(f"{status} {category:<20}: {age_count:3d}/{equipment_count:3d} (年数データ有り/全体)")
        
        if age_count > 0:
            # トップ3の設備を表示
            display_columns = ['設備id', '設備名', '現在年数']
            print(f"   トップ3設備:")
            top3 = equipment_with_age[display_columns].head(3)
            for _, row in top3.iterrows():
                print(f"     ID:{row['設備id']} {row['設備名']} ({row['現在年数']:.1f}年)")
            print()

if __name__ == "__main__":
    check_equipment_categories()