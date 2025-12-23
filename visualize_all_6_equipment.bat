@echo off
chcp 65001 >nul
echo =================================================================
echo            6台設備CBM学習結果可視化バッチ実行
echo =================================================================

echo.
echo 1/6: 薬注ポンプCP-500-5 (19.7年, 老朽化設備) - 可視化中...
python visualize_results.py --output_dir outputs_pump_265715 --equipment_id 265715 --measurement_id 260374 --analyze_dist
if %errorlevel% neq 0 (
    echo [ERROR] 薬注ポンプCP-500-5の可視化に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] 薬注ポンプCP-500-5可視化完了

echo.
echo 2/6: 冷却水ポンプCDP-A5 (3.0年, 新しい設備) - 可視化中...
python visualize_results.py --output_dir outputs_pump_137953 --equipment_id 137953 --measurement_id 166580 --analyze_dist
if %errorlevel% neq 0 (
    echo [ERROR] 冷却水ポンプCDP-A5の可視化に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] 冷却水ポンプCDP-A5可視化完了

echo.
echo 3/6: 薬注ポンプCP-500-3 (0.5年, 最新設備) - 可視化中...
python visualize_results.py --output_dir outputs_pump_519177 --equipment_id 519177 --measurement_id 416412 --analyze_dist
if %errorlevel% neq 0 (
    echo [ERROR] 薬注ポンプCP-500-3の可視化に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] 薬注ポンプCP-500-3可視化完了

echo.
echo 4/6: AHU-TSK-A-2 (15.6年, エアハンドリングユニット) - 可視化中...
python visualize_results.py --output_dir outputs_ahu_327240 --equipment_id 327240 --measurement_id 353609 --analyze_dist
if %errorlevel% neq 0 (
    echo [ERROR] AHU-TSK-A-2の可視化に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] AHU-TSK-A-2可視化完了

echo.
echo 5/6: R-1-3 (19.7年, 冷却器設備) - 可視化中...
python visualize_results.py --output_dir outputs_r13_265694 --equipment_id 265694 --measurement_id 258887 --analyze_dist
if %errorlevel% neq 0 (
    echo [ERROR] R-1-3の可視化に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] R-1-3可視化完了

echo.
echo 6/6: OAC-TSK-F-2 (17.7年, 外気処理機) - 可視化中...
python visualize_results.py --output_dir outputs_oac_322220 --equipment_id 322220 --measurement_id 344201 --analyze_dist
if %errorlevel% neq 0 (
    echo [ERROR] OAC-TSK-F-2の可視化に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] OAC-TSK-F-2可視化完了

echo.
echo =================================================================
echo            [COMPLETED] 全6台の可視化完了！
echo =================================================================
echo.
echo [RESULTS] 生成された可視化ファイル:
echo   - outputs_pump_265715  : 薬注ポンプCP-500-5 (19.7年)
echo   - outputs_pump_137953  : 冷却水ポンプCDP-A5 (3.0年)
echo   - outputs_pump_519177  : 薬注ポンプCP-500-3 (0.5年)
echo   - outputs_ahu_327240   : AHU-TSK-A-2 (15.6年)
echo   - outputs_r13_265694   : R-1-3 (19.7年)
echo   - outputs_oac_322220   : OAC-TSK-F-2 (17.7年)
echo.
echo 各ディレクトリには以下のグラフが生成されます:
echo   • training_history.png - 学習進行カーブ
echo   • transition_matrix.png - 状態遷移行列
echo   • aging_analysis.png - 設備老朽化分析
echo   • policy_evaluation.png - 政策評価
echo   • distribution_statistics.png - 収益分布統計
echo   • uncertainty_analysis.png - 不確実性分析
echo   • risk_profile.png - VaR/CVaR リスク分析
echo   • quantile_distributions.png - QR-DQN分位点分布
echo.
pause