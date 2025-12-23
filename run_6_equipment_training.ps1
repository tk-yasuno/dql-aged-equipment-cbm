# 6台設備CBM強化学習実行スクリプト
# 各設備の学習結果を個別ディレクトリに保存

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "            6台設備CBM強化学習バッチ実行"                           -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$equipment_list = @(
    @{name="薬注ポンプCP-500-5"; id=265715; measurement_id=260374; aging_factor=0.018; output="outputs_pump_265715"; age="19.7年"; description="老朽化設備"},
    @{name="冷却水ポンプCDP-A5"; id=137953; measurement_id=166580; aging_factor=0.005; output="outputs_pump_137953"; age="3.0年"; description="新しい設備"},
    @{name="薬注ポンプCP-500-3"; id=519177; measurement_id=416412; aging_factor=0.003; output="outputs_pump_519177"; age="0.5年"; description="最新設備"},
    @{name="AHU-TSK-A-2"; id=327240; measurement_id=353609; aging_factor=0.015; output="outputs_ahu_327240"; age="15.6年"; description="エアハンドリングユニット"},
    @{name="R-1-3"; id=265694; measurement_id=258887; aging_factor=0.018; output="outputs_r13_265694"; age="19.7年"; description="冷却器設備"},
    @{name="OAC-TSK-F-2"; id=322220; measurement_id=344201; aging_factor=0.015; output="outputs_oac_322220"; age="17.7年"; description="外気処理機"}
)

$start_time = Get-Date
$success_count = 0
$failed_equipment = @()

for ($i = 0; $i -lt $equipment_list.Count; $i++) {
    $equip = $equipment_list[$i]
    $current = $i + 1
    
    Write-Host ""
    Write-Host "$current/6: $($equip.name) ($($equip.age), $($equip.description))" -ForegroundColor Yellow
    Write-Host "      設備ID: $($equip.id) | 測定ID: $($equip.measurement_id) | 老朽化係数: $($equip.aging_factor)" -ForegroundColor Gray
    
    $cmd = "python train_cbm_dqn_v2.py --equipment_id $($equip.id) --measurement_id $($equip.measurement_id) --episodes 4000 --scenario balanced --aging_factor $($equip.aging_factor) --output_dir $($equip.output)"
    
    Write-Host "      実行中..." -ForegroundColor Cyan
    $result = Invoke-Expression $cmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      ✅ $($equip.name)完了" -ForegroundColor Green
        $success_count++
    } else {
        Write-Host "      ❌ $($equip.name)失敗 (終了コード: $LASTEXITCODE)" -ForegroundColor Red
        $failed_equipment += $equip.name
    }
}

$end_time = Get-Date
$duration = $end_time - $start_time

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "            学習結果サマリー"                                      -ForegroundColor Cyan  
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "実行時間: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor White
Write-Host "成功: $success_count/6台" -ForegroundColor Green

if ($failed_equipment.Count -gt 0) {
    Write-Host "失敗: $($failed_equipment.Count)台" -ForegroundColor Red
    Write-Host "失敗した設備: $($failed_equipment -join ', ')" -ForegroundColor Red
}

Write-Host ""
Write-Host "📊 学習結果保存先:" -ForegroundColor Cyan
foreach ($equip in $equipment_list) {
    if (Test-Path $equip.output) {
        Write-Host "  - $($equip.output)  : $($equip.name) ($($equip.age))" -ForegroundColor White
    } else {
        Write-Host "  - $($equip.output)  : $($equip.name) ($($equip.age)) [失敗]" -ForegroundColor Red
    }
}

if ($success_count -eq 6) {
    Write-Host ""
    Write-Host "🎉 全6台の学習が正常に完了しました！" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️  一部の設備で学習が失敗しました。ログを確認してください。" -ForegroundColor Yellow
}