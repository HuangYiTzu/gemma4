#!/usr/bin/env bash
# =====================================================================
# activate_env.sh — RMSNorm 驗證環境設定
# =====================================================================
# 用法： source activate_env.sh
#
# 這個腳本不建立、不啟用任何 venv/conda 環境 —— 目前這台機器上並沒有
# 專屬的 venv/conda 環境在用（2026-09-16 確認：Softmax\.venv 存在但是
# 空的，沒裝任何套件），所有套件（transformers/torch/accelerate/...）
# 都已經裝在系統 Python 3.12.6 裡。這裡只做兩件事：
#   1. 把系統 Python 的路徑排到 PATH 最前面，確保 `python`/`python3`
#      呼叫到的是裝好套件的那個直譯器，而不是 Windows 內建的
#      WindowsApps python.exe 空殼（那個沒裝任何套件，會誤導人以為
#      要重新安裝）。
#   2. 指定 HF_HOME 到現有已經有 gemma 模型快取的路徑，避免
#      transformers 因為 HF_HOME 沒設而改用別的預設路徑、觸發重新下載。
#
# 如果之後真的建立了專屬 venv（例如把套件裝進 Softmax\.venv 或這裡
# 建一個新的），把下面 PYTHON_BIN / VENV_ACTIVATE 改掉即可，其餘印出
# 確認訊息的邏輯不用動。
# =====================================================================

# ---- 系統 Python（已確認裝有 transformers/torch/accelerate/...） ----
PYTHON_DIR="C:\Users\eva huang\AppData\Local\Programs\Python\Python312"
PYTHON_BIN="${PYTHON_DIR}\python.exe"

if [ ! -f "$PYTHON_BIN" ]; then
    echo "[activate_env.sh] 錯誤：找不到 $PYTHON_BIN"
    echo "  這個路徑是 2026-09-16 用 'where.exe python' 實際查到的，"
    echo "  如果機器換過/路徑不對，請自行更新 PYTHON_DIR 這個變數。"
    return 1 2>/dev/null || exit 1
fi

export PATH="${PYTHON_DIR}:${PATH}"

# ---- HF_HOME：指到現有已有 gemma 快取的路徑，不建立新的空快取 ----
export HF_HOME="$HOME/.cache/huggingface"

GEMMA_CACHE="${HF_HOME}/hub/models--google--gemma-4-E2B-it-qat-mobile-transformers"
if [ ! -d "$GEMMA_CACHE" ]; then
    echo "[activate_env.sh] 警告：預期的 gemma 快取資料夾不存在："
    echo "  $GEMMA_CACHE"
    echo "  HF_HOME 還是照設，但如果之後跑腳本觸發下載，先檢查這裡。"
fi

# ---- 確認訊息 ----
echo "===================================================="
echo "[activate_env.sh] 環境設定完成"
echo "===================================================="
echo "Python        : $("$PYTHON_BIN" --version 2>&1)"
echo "Python路徑    : $PYTHON_BIN"
echo "HF_HOME       : $HF_HOME"
echo "Gemma快取     : $([ -d "$GEMMA_CACHE" ] && echo '存在 OK' || echo '找不到，見上方警告')"
echo "----------------------------------------------------"
"$PYTHON_BIN" -c "
import transformers, accelerate, torch, safetensors, numpy
print(f'transformers  : {transformers.__version__}')
print(f'accelerate    : {accelerate.__version__}')
print(f'torch         : {torch.__version__}')
print(f'safetensors   : {safetensors.__version__}')
print(f'numpy         : {numpy.__version__}')
"
echo "===================================================="
