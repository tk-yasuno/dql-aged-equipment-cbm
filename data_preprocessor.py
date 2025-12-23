"""
CBMデータ前処理スクリプト
温度データから2x2マルコフ遷移行列を推定

データソース:
- 設備諸元_実測値100以上.csv
- 測定値examples_3設備_測定項目_実測値_20251217.csv

処理内容:
1. 上限値Smax・下限値Sminを使って normal/anomalous を判定
2. 時系列データから2x2状態遷移行列を推定
3. 設備ごとの統計情報を出力
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import json


class CBMDataPreprocessor:
    """CBM測定データから状態遷移行列を推定するクラス"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.equipment_specs = None
        self.measurement_data = None
        self.installation_dates = None  # 設備の設置年月日データ
        
    def load_data(self):
        """CSVファイルを読み込む"""
        # 設備諸元の読み込み
        specs_path = self.data_dir / "設備諸元_実測値100以上.csv"
        self.equipment_specs = pd.read_csv(specs_path)
        print(f"✅ 設備諸元読み込み: {len(self.equipment_specs)} 行")
        
        # 測定値の読み込み
        measurements_path = self.data_dir / "測定値examples_3設備_測定項目_実測値_20251217.csv"
        self.measurement_data = pd.read_csv(measurements_path)
        print(f"✅ 測定値読み込み: {len(self.measurement_data)} 行")
        
        # 測定時刻をdatetime型に変換
        self.measurement_data['測定時刻'] = pd.to_datetime(
            self.measurement_data['測定時刻'], 
            format='mixed'
        )
        
        # 設備の設置年月日の読み込み
        installation_path = self.data_dir / "設備の設置年月日.csv"
        if installation_path.exists():
            self.installation_dates = pd.read_csv(installation_path)
            # カラム名を確認・統一
            if '設備年月日' in self.installation_dates.columns:
                self.installation_dates['設備年月日'] = self.installation_dates['設備年月日']
            
            # 設備年月日をdatetime型に変換（複数の日付フォーマットに対応）
            def parse_date(date_str):
                if pd.isna(date_str):
                    return pd.NaT
                
                date_str = str(date_str).strip()
                
                # 複数のフォーマットを試す
                formats_to_try = [
                    '%Y-%m-%d',    # 1881-03-01
                    '%Y/%m/%d',    # 9999/12/31
                    '%Y-%m',       # 1881-03
                    '%Y/%m',       # 9999/12
                ]
                
                for fmt in formats_to_try:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                
                # 最後の手段：pandas自動判定
                try:
                    return pd.to_datetime(date_str, errors='coerce')
                except:
                    return pd.NaT
            
            self.installation_dates['設備年月日'] = self.installation_dates['設備年月日'].apply(parse_date)
            
            # 有効範囲のデータのみ残す（1900年-現在）
            current_year = pd.Timestamp.now().year
            valid_dates = (
                (self.installation_dates['設備年月日'] >= '1900-01-01') &
                (self.installation_dates['設備年月日'] <= f'{current_year}-12-31') &
                (~self.installation_dates['設備年月日'].isna())
            )
            
            print(f"   全データ: {len(self.installation_dates)} 行")
            print(f"   有効な日付: {valid_dates.sum()} 行")
            
            self.installation_dates = self.installation_dates[valid_dates]
            print(f"✅ 設備設置年月日読み込み: {len(self.installation_dates)} 行")
            
            if len(self.installation_dates) > 0:
                print(f"   有効な設備ID範囲: {self.installation_dates['設備ID'].min()} - {self.installation_dates['設備ID'].max()}")
            else:
                print("   ⚠️ 有効なデータが見つかりません")
        else:
            print(f"⚠️ 設備設置年月日ファイルが見つかりません: {installation_path}")
        
    def get_available_equipment_with_age(self, equipment_class: str = "機械設備") -> pd.DataFrame:
        """設備年数データがある利用可能な設備一覧を取得"""
        if self.installation_dates is None:
            print("⚠️ 設備設置年月日データが読み込まれていません")
            return pd.DataFrame()
        
        # 基本的な設備一覧を取得
        basic_equipment = self.get_available_equipment(equipment_class)
        
        # 設置年月日データがある設備のIDリスト
        available_equipment_ids = set(self.installation_dates['設備ID'].tolist())
        
        # 設備年数データがある設備のみをフィルタ
        equipment_with_age = basic_equipment[
            basic_equipment['設備id'].isin(available_equipment_ids)
        ].copy()
        
        # 設置年月日情報を追加
        if len(equipment_with_age) > 0:
            equipment_with_age = equipment_with_age.merge(
                self.installation_dates[['設備ID', '設備年月日']],
                left_on='設備id',
                right_on='設備ID',
                how='left'
            )
            
            # 現在の設備年数を計算
            current_time = pd.Timestamp.now()
            equipment_with_age['現在年数'] = (
                (current_time - equipment_with_age['設備年月日']).dt.days / 365.25
            )
            
            # ソート
            equipment_with_age = equipment_with_age.sort_values('総測定回数', ascending=False)
        
        return equipment_with_age
    
    def get_available_equipment(self, equipment_class: str = "機械設備") -> pd.DataFrame:
        """利用可能な設備一覧を取得"""
        filtered = self.equipment_specs[
            self.equipment_specs['設備分類'] == equipment_class
        ]
        
        # 設備ごとにグループ化
        equipment_summary = filtered.groupby(['設備id', '設備名']).agg({
            '測定項目id': 'count',
            '測定回数': 'sum'
        }).reset_index()
        equipment_summary.columns = ['設備id', '設備名', '測定項目数', '総測定回数']
        
        return equipment_summary.sort_values('総測定回数', ascending=False)
    
    def get_measurement_items(self, equipment_id: int) -> pd.DataFrame:
        """特定設備の測定項目一覧を取得"""
        items = self.equipment_specs[
            self.equipment_specs['設備id'] == equipment_id
        ][['測定項目id', '測定指標', '測定回数', '最新の実測値']]
        
        return items.sort_values('測定回数', ascending=False)
    
    def get_equipment_age(self, equipment_id: int, measurement_time: pd.Timestamp) -> Optional[float]:
        """設備の経過年数を計算
        
        Args:
            equipment_id: 設備ID
            measurement_time: 測定日時
            
        Returns:
            経過年数（float）。データがない場合はNone
        """
        if self.installation_dates is None:
            return None
            
        equipment_installation = self.installation_dates[
            self.installation_dates['設備ID'] == equipment_id
        ]
        
        if len(equipment_installation) == 0:
            return None
            
        installation_date = equipment_installation['設備年月日'].iloc[0]
        if pd.isna(installation_date):
            return None
            
        # 経過年数を計算（年単位で小数も含む）
        age_years = (measurement_time - installation_date).days / 365.25
        return max(0, age_years)  # 負の値は0にクリップ
    
    def extract_timeseries(
        self, 
        equipment_id: int, 
        measurement_id: int,
        include_age: bool = True
    ) -> pd.DataFrame:
        """特定の設備・測定項目の時系列データを抽出
        
        Args:
            equipment_id: 設備ID
            measurement_id: 測定項目ID
            include_age: 経過年数を含めるか
        """
        df = self.measurement_data[
            (self.measurement_data['設備id'] == equipment_id) &
            (self.measurement_data['状態測定項目id'] == measurement_id)
        ].copy()
        
        # 時間でソート
        df = df.sort_values('測定時刻').reset_index(drop=True)
        
        # 経過年数を追加
        if include_age and self.installation_dates is not None:
            df['設備経過年数'] = df['測定時刻'].apply(
                lambda x: self.get_equipment_age(equipment_id, x)
            )
        
        return df
    
    def label_states(self, df: pd.DataFrame, k_sigma: float = 2.0) -> pd.DataFrame:
        """CSVに含まれる上限値Smax・下限値Sminに基づいて状態をラベリング
        
        Args:
            df: 時系列データフレーム（上限値Smax, 下限値Smin, 実測値を含む）
            k_sigma: 統計的閾値計算時の標準偏差の係数（デフォルト: 2.0）
        
        Returns:
            状態カラム(condition)を追加したDataFrame
            condition = 0: normal (Smin <= 実測値 <= Smax)
            condition = 1: anomalous (それ以外)
        """
        df = df.copy()
        
        # 閾値の取得（CSVから）
        Smin = df['下限値Smin'].iloc[0]
        Smax = df['上限値Smax'].iloc[0]
        
        # 閾値が欠損している場合は統計的に計算
        if pd.isna(Smin) or pd.isna(Smax):
            mu = df['実測値'].mean()
            sigma = df['実測値'].std()
            
            if pd.isna(Smin):
                Smin = mu - k_sigma * sigma
                print(f"   ℹ️ 下限値Sminが欠損のため統計的に計算: {Smin:.2f} (μ - {k_sigma}σ)")
            
            if pd.isna(Smax):
                Smax = mu + k_sigma * sigma
                print(f"   ℹ️ 上限値Smaxが欠損のため統計的に計算: {Smax:.2f} (μ + {k_sigma}σ)")
            
            # DataFrameに統計的閾値を設定
            df['下限値Smin'] = Smin
            df['上限値Smax'] = Smax
        
        # 状態判定: normal(0) / anomalous(1)
        df['condition'] = np.where(
            (df['実測値'] >= Smin) & (df['実測値'] <= Smax),
            0,  # normal
            1   # anomalous
        )
        
        return df
    
    def estimate_transition_matrix(self, df: pd.DataFrame) -> Dict:
        """2x2マルコフ状態遷移行列を推定
        
        Args:
            df: conditionカラムを含むDataFrame
        
        Returns:
            遷移行列と統計情報を含む辞書
        """
        conditions = df['condition'].values
        
        # 遷移回数をカウント
        N00 = np.sum((conditions[:-1] == 0) & (conditions[1:] == 0))  # normal → normal
        N01 = np.sum((conditions[:-1] == 0) & (conditions[1:] == 1))  # normal → anomalous
        N10 = np.sum((conditions[:-1] == 1) & (conditions[1:] == 0))  # anomalous → normal
        N11 = np.sum((conditions[:-1] == 1) & (conditions[1:] == 1))  # anomalous → anomalous
        
        # 遷移確率行列を計算
        total_from_normal = N00 + N01
        total_from_anomalous = N10 + N11
        
        if total_from_normal == 0:
            p00, p01 = 0.0, 0.0
        else:
            p00 = N00 / total_from_normal
            p01 = N01 / total_from_normal
        
        if total_from_anomalous == 0:
            p10, p11 = 0.0, 0.0
        else:
            p10 = N10 / total_from_anomalous
            p11 = N11 / total_from_anomalous
        
        transition_matrix = np.array([
            [p00, p01],  # from normal
            [p10, p11]   # from anomalous
        ], dtype=np.float32)
        
        # 統計情報
        n_total = len(conditions)
        n_normal = np.sum(conditions == 0)
        n_anomalous = np.sum(conditions == 1)
        
        stats = {
            'transition_matrix': transition_matrix.tolist(),
            'transition_counts': {
                'N00': int(N00), 'N01': int(N01),
                'N10': int(N10), 'N11': int(N11)
            },
            'state_counts': {
                'normal': int(n_normal),
                'anomalous': int(n_anomalous),
                'total': int(n_total)
            },
            'state_distribution': {
                'normal_ratio': float(n_normal / n_total),
                'anomalous_ratio': float(n_anomalous / n_total)
            },
            'thresholds': {
                'Smax': float(df['上限値Smax'].iloc[0]),
                'Smin': float(df['下限値Smin'].iloc[0])
            },
            'value_stats': {
                'mean': float(df['実測値'].mean()),
                'std': float(df['実測値'].std()),
                'min': float(df['実測値'].min()),
                'max': float(df['実測値'].max())
            }
        }
        
        return stats
    
    def process_equipment(
        self,
        equipment_id: int,
        measurement_id: int,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """特定設備の測定データを処理して遷移行列を推定
        
        Args:
            equipment_id: 設備ID
            measurement_id: 測定項目ID
            output_dir: 結果の保存先ディレクトリ（指定すればJSONとCSVを出力）
        
        Returns:
            遷移行列と統計情報
        """
        # データ抽出
        df = self.extract_timeseries(equipment_id, measurement_id)
        
        if len(df) == 0:
            raise ValueError(f"データが見つかりません: 設備ID={equipment_id}, 測定項目ID={measurement_id}")
        
        print(f"\n📊 データ抽出: {len(df)} データポイント")
        
        # 状態ラベリング
        df = self.label_states(df)
        
        # 遷移行列推定
        stats = self.estimate_transition_matrix(df)
        
        # 老朽化パラメータ推定
        age_params = self.estimate_age_adjusted_parameters(df)
        stats['age_analysis'] = age_params
        
        # 設備情報を追加
        equipment_info = self.equipment_specs[
            (self.equipment_specs['設備id'] == equipment_id) &
            (self.equipment_specs['測定項目id'] == measurement_id)
        ].iloc[0]
        
        stats['equipment_id'] = int(equipment_id)
        stats['measurement_id'] = int(measurement_id)
        stats['equipment_name'] = equipment_info['設備名']
        stats['measurement_name'] = equipment_info['測定指標']
        
        # 結果表示
        print(f"\n✅ 設備: {stats['equipment_name']}")
        print(f"✅ 測定項目: {stats['measurement_name']}")
        print(f"✅ 閾値: Smin={stats['thresholds']['Smin']}, Smax={stats['thresholds']['Smax']}")
        print(f"\n📈 状態分布:")
        print(f"  - Normal: {stats['state_counts']['normal']} ({stats['state_distribution']['normal_ratio']*100:.1f}%)")
        print(f"  - Anomalous: {stats['state_counts']['anomalous']} ({stats['state_distribution']['anomalous_ratio']*100:.1f}%)")
        print(f"\n🔄 状態遷移行列 (2x2):")
        P = np.array(stats['transition_matrix'])
        print(f"  [[{P[0,0]:.4f}, {P[0,1]:.4f}],  # normal → [normal, anomalous]")
        print(f"   [{P[1,0]:.4f}, {P[1,1]:.4f}]]  # anomalous → [normal, anomalous]")
        
        # 老朽化分析結果の表示
        if age_params['age_available']:
            print(f"\n🏗️ 老朽化分析:")
            print(f"  - 設備年数範囲: {age_params['age_range'][0]:.1f} - {age_params['age_range'][1]:.1f} 年")
            print(f"  - 年数と異常率の相関: {age_params['overall_age_correlation']:.3f}")
            if age_params['degradation_trend']:
                trend = age_params['degradation_trend']
                print(f"  - 劣化トレンド: 年間+{trend['slope']*100:.2f}%の異常率増加 (相関: {trend['correlation']:.3f})")
        else:
            print(f"\n⚠️ 老朽化分析: {age_params['message']}")
        
        # ファイル保存
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # JSON保存
            json_path = output_dir / f"equipment_{equipment_id}_measurement_{measurement_id}_stats.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"\n💾 統計情報を保存: {json_path}")
            
            # 時系列CSV保存
            csv_path = output_dir / f"equipment_{equipment_id}_measurement_{measurement_id}_timeseries.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 時系列データを保存: {csv_path}")
        
        return stats
    
    def estimate_age_adjusted_parameters(self, df: pd.DataFrame) -> Dict:
        """設備年数を考慮した劣化パラメータを推定
        
        Args:
            df: 設備経過年数カラムを含むDataFrame
            
        Returns:
            年数別の統計情報と劣化トレンドを含む辞書
        """
        if '設備経過年数' not in df.columns or df['設備経過年数'].isna().all():
            return {
                'age_available': False,
                'message': '設備経過年数データが利用できません'
            }
        
        # 経過年数の有効なデータのみ使用
        valid_age_df = df.dropna(subset=['設備経過年数'])
        
        if len(valid_age_df) == 0:
            return {
                'age_available': False,
                'message': '有効な設備経過年数データがありません'
            }
        
        # 年数区分でグループ化（5年刻み）
        max_age = int(valid_age_df['設備経過年数'].max())
        age_bins = list(range(0, max_age + 6, 5))
        valid_age_df['age_group'] = pd.cut(valid_age_df['設備経過年数'], bins=age_bins, right=False)
        
        age_stats = []
        for age_group, group_data in valid_age_df.groupby('age_group'):
            if len(group_data) < 5:  # 十分なデータがない場合はスキップ
                continue
                
            anomaly_rate = group_data['condition'].mean()  # 異常率
            mean_age = group_data['設備経過年数'].mean()
            
            age_stats.append({
                'age_group': str(age_group),
                'mean_age': mean_age,
                'data_points': len(group_data),
                'anomaly_rate': anomaly_rate,
                'mean_measurement': group_data['実測値'].mean(),
                'std_measurement': group_data['実測値'].std()
            })
        
        # 劣化トレンドの線形回帰
        degradation_trend = None
        if len(age_stats) >= 2:
            ages = [s['mean_age'] for s in age_stats]
            anomaly_rates = [s['anomaly_rate'] for s in age_stats]
            
            # 簡単な線形回帰
            age_array = np.array(ages)
            rate_array = np.array(anomaly_rates)
            
            A = np.vstack([age_array, np.ones(len(age_array))]).T
            slope, intercept = np.linalg.lstsq(A, rate_array, rcond=None)[0]
            
            degradation_trend = {
                'slope': slope,  # 年当たりの異常率増加
                'intercept': intercept,
                'correlation': np.corrcoef(ages, anomaly_rates)[0, 1] if len(ages) > 1 else 0.0
            }
        
        return {
            'age_available': True,
            'age_range': [valid_age_df['設備経過年数'].min(), valid_age_df['設備経過年数'].max()],
            'age_stats_by_group': age_stats,
            'degradation_trend': degradation_trend,
            'overall_age_correlation': np.corrcoef(
                valid_age_df['設備経過年数'], 
                valid_age_df['condition']
            )[0, 1] if len(valid_age_df) > 1 else 0.0
        }


def main():
    """メイン処理: 年数データがある設備の前処理デモ"""
    
    # データディレクトリ
    data_dir = Path(__file__).parent.parent / "data" / "private_benchmark"
    output_dir = Path(__file__).parent / "preprocessed_data"
    
    # プリプロセッサ初期化
    preprocessor = CBMDataPreprocessor(data_dir)
    preprocessor.load_data()
    
    # 年数データ付き設備一覧を取得（空調設備を対象）
    print("\n" + "="*60)
    print("📋 年数データ付き利用可能な空調設備一覧")
    print("="*60)
    equipment_list_with_age = preprocessor.get_available_equipment_with_age("空調設備")
    
    if len(equipment_list_with_age) == 0:
        print("⚠️ 年数データがある空調設備が見つかりませんでした")
        print("\n通常の空調設備一覧:")
        equipment_list = preprocessor.get_available_equipment("空調設備")
        print(equipment_list.head().to_string(index=False))
        return
    
    print(f"年数データがある空調設備数: {len(equipment_list_with_age)}")
    display_columns = ['設備id', '設備名', '測定項目数', '総測定回数', '現在年数']
    print(equipment_list_with_age[display_columns].head(10).to_string(index=False))
    
    # 最も測定回数が多い設備を選択
    target_equipment = equipment_list_with_age.iloc[0]
    equipment_id = target_equipment['設備id']
    equipment_age = target_equipment['現在年数']
    
    print(f"\n" + "="*60)
    print(f"🎯 対象空調設備: {target_equipment['設備名']} [ID: {equipment_id}]")
    print(f"   設備年数: {equipment_age:.1f} 年")
    print("="*60)
    
    # 測定項目を確認
    measurement_items = preprocessor.get_measurement_items(equipment_id)
    print(measurement_items.head().to_string(index=False))
    
    # 最も測定回数が多い項目を選択
    measurement_id = measurement_items.iloc[0]['測定項目id']
    
    print(f"\n" + "="*60)
    print("🎯 データ処理実行")
    print("="*60)
    
    # データ処理と遷移行列推定
    stats = preprocessor.process_equipment(
        equipment_id=equipment_id,
        measurement_id=measurement_id,
        output_dir=output_dir
    )
    
    print(f"\n" + "="*60)
    print("✅ 前処理完了 - 年数データ付き空調設備での処理成功")
    print(f"対象空調設備ID: {equipment_id}, 測定項目ID: {measurement_id}")
    print(f"設備年数: {equipment_age:.1f} 年")
    print("="*60)
    
    print("\n" + "="*60)
    print("✅ 前処理完了")
    print("="*60)


if __name__ == "__main__":
    main()
