@echo off
chcp 65001 >nul
echo =================================================================
echo            6台設備CBM強化学習バッチ実行 (4000エピソード)
echo =================================================================

echo.
echo 1/6: 薬注ポンプCP-500-5 (19.7年, 老朽化設備)
python train_cbm_dqn_v2.py --equipment_id 265715 --measurement_id 260374 --episodes 4000 --scenario balanced --aging_factor 0.018 --output_dir outputs_pump_265715
if %errorlevel% neq 0 (
    echo [ERROR] 薬注ポンプCP-500-5の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] 薬注ポンプCP-500-5完了

echo.
echo 2/6: 冷却水ポンプCDP-A5 (3.0年, 新しい設備)
python train_cbm_dqn_v2.py --equipment_id 137953 --measurement_id 166580 --episodes 4000 --scenario balanced --aging_factor 0.005 --output_dir outputs_pump_137953
if %errorlevel% neq 0 (
    echo [ERROR] 冷却水ポンプCDP-A5の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] 冷却水ポンプCDP-A5完了

echo.
echo 3/6: 薬注ポンプCP-500-3 (0.5年, 最新設備)
python train_cbm_dqn_v2.py --equipment_id 519177 --measurement_id 416412 --episodes 4000 --scenario balanced --aging_factor 0.003 --output_dir outputs_pump_519177
if %errorlevel% neq 0 (
    echo [ERROR] 薬注ポンプCP-500-3の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] 薬注ポンプCP-500-3完了

echo.
echo 4/6: AHU-TSK-A-2 (15.6年, エアハンドリングユニット)
python train_cbm_dqn_v2.py --equipment_id 327240 --measurement_id 353609 --episodes 4000 --scenario balanced --aging_factor 0.015 --output_dir outputs_ahu_327240
if %errorlevel% neq 0 (
    echo [ERROR] AHU-TSK-A-2の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] AHU-TSK-A-2完了

echo.
echo 5/6: R-1-3 (19.7年, 冷却器設備)
python train_cbm_dqn_v2.py --equipment_id 265694 --measurement_id 258887 --episodes 4000 --scenario balanced --aging_factor 0.018 --output_dir outputs_r13_265694
if %errorlevel% neq 0 (
    echo [ERROR] R-1-3の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] R-1-3完了

echo.
echo 6/6: OAC-TSK-F-2 (17.7年, 外気処理機)
python train_cbm_dqn_v2.py --equipment_id 322220 --measurement_id 344201 --episodes 4000 --scenario balanced --aging_factor 0.015 --output_dir outputs_oac_322220
if %errorlevel% neq 0 (
    echo [ERROR] OAC-TSK-F-2の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] OAC-TSK-F-2完了

echo.
echo =================================================================
echo            [COMPLETED] 全6台の学習完了！(4000エピソード)
echo =================================================================
echo.
echo [RESULTS] 学習結果保存先:
echo   - outputs_pump_265715  : 薬注ポンプCP-500-5 (19.7年)
echo   - outputs_pump_137953  : 冷却水ポンプCDP-A5 (3.0年)
echo   - outputs_pump_519177  : 薬注ポンプCP-500-3 (0.5年)
echo   - outputs_ahu_327240   : AHU-TSK-A-2 (15.6年)
echo   - outputs_r13_265694   : R-1-3 (19.7年)
echo   - outputs_oac_322220   : OAC-TSK-F-2 (17.7年)
echo.
pause