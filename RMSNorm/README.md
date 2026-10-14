# RMSNorm 驗證工作區

這個資料夾整理了 Gemma-4-E2B-it-qat-mobile-transformers 的 RMSNorm 驗證相關程式碼：golden model、QAT模型驗證腳本、以及之後的wikitext校準腳本。

**這個資料夾不含任何模型權重或Python套件。** 所有東西都指向這台機器上已經存在的東西：
- Python套件（transformers/torch/accelerate/safetensors/numpy）已經裝在系統Python 3.12.6裡
- Gemma模型快取已經存在於 `~/.cache/huggingface`

執行任何腳本前，請先在這個資料夾（或子資料夾）下跑：

```bash
source activate_env.sh
```

這個指令只會設定PATH與HF_HOME環境變數指向現有的Python與快取，**不會**觸發任何pip install或模型下載。

## 資料夾結構

| 資料夾 | 用途 |
|---|---|
| `golden_model/` | `rmsnorm_golden_model.py` — RMSNorm的整數golden model骨架，串接了input_layernorm、q/k/v_norm、post_attention_layernorm、pre/post_feedforward_layernorm、post_per_layer_input_norm等呼叫點 |
| `qat_verification/` | `extract_real_rmsnorm_activation.py`（用hook從真實QAT模型抓RMSNorm前後的activation）、`rmsnorm_verify_pre_post_mlp.py`（拿抓到的真實值跟golden model做浮點/整數三方比對） |
| `calibration/` | 之後放wikitext校準相關腳本（目前是空的） |
| `outputs/` | 跑腳本產生的中間`.npy`檔案（例如`real_residual_pre_layer15.npy`等），建議把`extract_real_rmsnorm_activation.py`的輸出路徑導到這裡 |

## 驗證環境設定是否正確

跑完`source activate_env.sh`後，應該會看到Python版本、HF_HOME路徑、各套件版本的確認訊息。也可以手動再驗證一次：

```bash
python -c "import transformers; print(transformers.__version__)"
```

如果印出版本號（不是錯誤、也沒有觸發下載進度條），代表環境設定正確，可以直接跑`qat_verification/`底下的腳本，不需要額外安裝任何東西。

## 使用順序

1. `source activate_env.sh`
2. 跑 `qat_verification/extract_real_rmsnorm_activation.py`，抓真實QAT模型的RMSNorm前後activation（存成`.npy`，建議搬到`outputs/`）
3. 跑 `qat_verification/rmsnorm_verify_pre_post_mlp.py`，比對golden model（`golden_model/rmsnorm_golden_model.py`）輸出跟真實模型輸出的誤差
4. `calibration/`底下的腳本（之後補上）用來對更大樣本的wikitext資料做S_RES/S_Q/S_K/S_V等scale的校準
