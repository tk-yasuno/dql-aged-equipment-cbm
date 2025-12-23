@echo off
chcp 65001 >nul
echo =================================================================
echo     AHU系追加設備2台CBM強化学習バッチ実行 (2000エピソード)
echo     既存のAHU-TSK-A-2の優秀実績を検証
echo =================================================================

echo.
echo 1/2: AHU-TSK-F-4 (14.2年, プレフィルタ差圧) - 測定項目ID: 353407
python train_cbm_dqn_v2.py --equipment_id 327280 --measurement_id 353407 --episodes 2000 --scenario balanced --equipment_age 14.2 --aging_factor 0.0151 --output_dir outputs_ahu_327280_dp1
if %errorlevel% neq 0 (
    echo [ERROR] AHU-TSK-F-4 (差圧1)の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] AHU-TSK-F-4 (差圧1)完了

echo.
echo 2/2: AHU-TSK-F-4 (14.2年, プレフィルタ差圧) - 測定項目ID: 353409  
python train_cbm_dqn_v2.py --equipment_id 327280 --measurement_id 353409 --episodes 2000 --scenario balanced --equipment_age 14.2 --aging_factor 0.0151 --output_dir outputs_ahu_327280_dp2
if %errorlevel% neq 0 (
    echo [ERROR] AHU-TSK-F-4 (差圧2)の学習に失敗しました
    pause
    exit /b %errorlevel%
)
echo [OK] AHU-TSK-F-4 (差圧2)完了

echo.
echo =================================================================
echo            AHU系追加2台学習完了！
echo =================================================================
echo 既存実績：AHU-TSK-A-2 (15.6年) = +73.88 (最優秀)
echo 新規候補：AHU-TSK-F-4 (14.2年) = 結果確認中
echo 
echo 出力ディレクトリ:
echo   - outputs_ahu_327280_dp1 (測定項目353407)
echo   - outputs_ahu_327280_dp2 (測定項目353409)
echo =================================================================
pause