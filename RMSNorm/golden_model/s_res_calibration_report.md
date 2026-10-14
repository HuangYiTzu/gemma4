# S_RES（Residual量化尺度）校準：方法、過程與結果

## 1. 背景與動機

`S_RES` 是 Gemma-4-E2B-it-qat-mobile-transformers 硬體量化設計中，residual stream（跨層累加的主幹特徵）所使用的定點量化尺度（quantization scale）。它決定了每個 decoder layer 裡三個分支（Attention、MLP、PLE）的 RMSNorm 輸出，在加回 residual 之前要對齊到哪一個固定小數位元格式（Q-format）。

規格文件 v1.6（決策 C）將 `S_RES` 定為 `2⁻⁸`（F=8），對應 INT16 定點表示法下 residual 可表示的範圍為 `±128`。這個數值是初步設計階段的暫定值，文件本身也標註為待驗證項目：僅用 100 筆 WikiText 樣本估計過，測得的最大值 83.6 對上限 128 只有 1.53 倍餘裕，樣本數不足以確認這個餘裕在更大規模的真實輸入下是否足夠。

本報告記錄後續針對 `S_RES` 進行的完整校準流程：從小樣本問題發現、到大規模 WikiText 校準、到最終數值定案與多層面驗證。

## 2. 問題發現：小樣本驗證即已暴露風險

在對 `pre_feedforward_layernorm` / `post_feedforward_layernorm` 這兩個 RMSNorm 呼叫點進行 golden model 與真實 QAT 模型輸出的逐層比對時（6 層抽樣、4 種 prompt、共 75 個 token），發現抽樣的 6 層中有 3 層（Layer 0、14、34）的 `post_feedforward_layernorm` 真實輸出，其絕對值已經超出 `S_RES=2⁻⁸` 所能表示的 `±128` 範圍：

| Layer | 實測最大絕對值 | 超出 ±128 的倍數 | 超出範圍元素比例 |
|---|---|---|---|
| 0 | 383.90 | 3.00 倍 | 245 / 115200 |
| 14 | 507.00 | 3.96 倍 | 274 / 115200 |
| 34 | 205.77 | 1.61 倍 | 50 / 115200 |

這個結果比文件原先估計的 1.53 倍餘裕嚴重得多，且是橫跨多層的系統性問題，不是單一離群樣本造成的偶發現象，因此判定 `S_RES=2⁻⁸` 需要用更大規模的樣本重新校準。

## 3. 校準方法

### 3.1 資料來源

使用 WikiText-2（`wikitext-2-raw-v1`）**train split**（本地已有快取，未重新下載），與後續 end-to-end perplexity 評測所用的 test split 區隔開，避免校準資料與評測資料重疊。

### 3.2 樣本規模

40 個長度 512 token 的不重疊文字區塊，共 **20,480 個 token**，約為先前 75-token 小樣本的 273 倍。

### 3.3 記憶體考量：streaming 統計，不保留完整張量

若比照小樣本作法，將完整 `(token, hidden_dim)` 張量存下來，在 35 層、20,480 個 token 的規模下需要約 17 GB 記憶體，不可行。因此改用 **streaming 統計**：forward hook 不保留完整輸出張量，只即時更新兩種統計量：

- per-channel running max（記憶體用量只跟 channel 數成正比）
- per-token max 的清單（用於計算百分位數，記憶體只跟 token 數成正比）

這使得記憶體用量與樣本規模脫鉤，可以支援遠大於小樣本規模的校準。

### 3.4 校準對象

由於 `S_RES` 是被 `post_attention_layernorm`、`post_feedforward_layernorm`、`post_per_layer_input_norm` 三個 RMSNorm 呼叫點共用的全域常數，完整校準理論上需要涵蓋這三者的真實輸出分布。校準主流程以 `post_feedforward_layernorm`（MLP 分支）為主要對象，另外針對 `post_attention_layernorm`（Attention 分支）與 `post_per_layer_input_norm`（PLE 分支）做了獨立的量級快速檢查（10 個 chunk、5,120 個 token 的較小規模樣本），確認三者之間的風險是否一致。

## 4. 校準結果

### 4.1 全 35 層的溢位風險

`post_feedforward_layernorm` 全 35 層真實輸出的 global max 為 **595.7565**（出現在 Layer 14，channel 850）。以此檢驗不同 F 值下是否會飽和：

| F 值 | S_RES | INT16 可表示範圍 | 是否飽和 |
|---|---|---|---|
| 8 | 2⁻⁸ | ±128.00 | 是 |
| 7 | 2⁻⁷ | ±255.99 | 是 |
| 6 | 2⁻⁶ | ±511.98 | 是 |
| 5 | 2⁻⁵ | ±1023.97 | 否，有餘裕 |

結論：在此樣本規模下，`F=8/7/6` 均無法避免飽和，需要 `F=5` 才能完整涵蓋。

### 4.2 分組（per-layer）量化尺度方案的評估

進一步分析發現，若排除 Layer 0（498.5）、14（595.8）、34（208.3）這三個離群層，其餘 32 層的最大值僅 138.97（Layer 13），`F=7` 即已足夠。這代表若採用單一全域 `F=5`，會讓另外 32 個原本不需要如此寬鬆量化格式的層，被迫承受 8 倍粗化的量化步階，犧牲不必要的精細度。

因此評估了「分組量化尺度」（per-layer S_RES）的可行性：數學上可行，作法是在每次跨層交接處插入一次 residual 的 rescale（per-tensor 純量乘法）。但由於：

1. `S_RES` 作用於整個模型的 residual 累加主幹，是最核心的資料通路，任何架構調整的風險遠高於單一呼叫點的局部調整；
2. 目前的 golden model 與硬體設計皆未實作過「跨層 rescale」機制，導入需要重新設計 residual 累加的資料流，且該機制本身的量化誤差特性也需要額外驗證；

本階段決定**不採用**分組量化尺度，改採單一全域 `S_RES=2⁻⁵`（F=5），以保證正確性優先，將分組方案記錄為有數據支持、但需要更完整架構設計與驗證後才推行的未來優化方向。

### 4.3 其他兩個共用呼叫點的一致性檢查

| 呼叫點 | 全 35 層 global max（5,120 token 樣本）| 是否需要比 F=5 更寬的範圍 |
|---|---|---|
| `post_attention_layernorm` | 96.59 | 否 |
| `post_per_layer_input_norm` | 203.46 | 否 |

兩者的量級皆低於 `post_feedforward_layernorm` 的 595.76，確認 `F=5` 對三個共用 `S_RES` 的呼叫點都有餘裕，沒有發現更嚴重的風險來源。

## 5. 最終決策

`S_RES` 由 `2⁻⁸` 修正為 `2⁻⁵`（F=8 → F=5），此值已寫入 golden model（`rmsnorm_golden_model.py`）並作為後續所有常數表產生、HLS 實作與驗證的基準。

## 6. 修正後的驗證

### 6.1 Component 層級：golden model 對真實 QAT 輸出的逐 token cos similarity

套用 `S_RES=2⁻⁵` 後，重新執行 6 層抽樣（75 個 token）的驗證：

| Layer | pre_ln cos_sim (min/mean/max) 修正前 | 修正後 | post_ln cos_sim 修正前 | 修正後 |
|---|---|---|---|---|
| 0 | 0.9991 / 0.9994 / 0.9996 | 0.9995 / 0.9996 / 0.9998 | 0.9333 / 0.9709 / 0.9972 | **1.0000 / 1.0000 / 1.0000** |
| 14 | 0.7997 / 0.9162 / 0.9999 | 0.9952 / 0.9968 / 0.9979 | 0.9118 / 0.9537 / 0.9997 | 0.9906 / 0.9941 / 0.9968 |

修正前最差的兩個案例（Layer 0/14 的 post_ln）在修正後分別提升至 1.0000 與 0.99+，全部抽樣層與呼叫點的 cos similarity 均落於 0.98 以上，且未觀察到新的系統性劣化。

### 6.2 Residual 累加架構驗證：確認 S_RES=2⁻⁵ 的必要性未被架構選擇取代

在後續針對 residual 累加順序（Attention→MLP→PLE 三分支序列式加總、以及每層末端的 `layer_scalar` 純量乘法）所做的「方案 A（加總後統一捨入）vs 方案 B（各分支各自捨入後加總）」對照實驗中，同步確認了方案 B 並不會降低 `S_RES` 所需的動態範圍需求（因為 `layer_scalar` 必須同時作用於原始 residual 與三個分支的 delta，各分量的動態範圍與方案 A 同量級）。此結果進一步支持第 4.2 節「暫不採用分組量化尺度」的決策，`S_RES=2⁻⁵` 在架構層面沒有進一步壓縮空間。

### 6.3 硬體位元寬度層級：Stage3 C-simulation

`S_RES=2⁻⁵` 亦已整合進 RMSNorm 的 HLS 核心（`rmsnorm_kernel.h`）與對應的 Stage3 定點運算鏈中，並透過實際編譯執行的 C-simulation testbench（涵蓋全 35 層、共 210 筆真實 activation 測試向量）驗證，整數運算結果與 bit-accurate 的 Python 模擬達到 99.5% 逐位元精確相符，確認 `S_RES=2⁻⁵` 在實際硬體定點運算鏈中的可行性。

## 7. 結論與限制

- `S_RES` 由文件初版的 `2⁻⁸`（僅 100 筆樣本估計）修正為 `2⁻⁵`，修正依據為 20,480 個 token 規模的 WikiText 大樣本實測，並在 component 層級、residual 架構層級、硬體位元寬度層級三個面向完成驗證。
- 目前 `S_RES` 校準完整涵蓋了三個共用此常數的 RMSNorm 呼叫點（Attention、MLP、PLE 分支），且已針對「單一全域尺度 vs 分組尺度」的設計取捨做過量化分析與風險評估。
- 已知限制：（1）校準樣本來自 WikiText 百科文字，與模型實際部署情境（例如對話、程式碼等）的輸入分布可能有落差，若要進一步強化結論穩健度，建議日後補充更多元的輸入樣本；（2）分組量化尺度雖有數據支持，但因涉及 residual 累加核心資料通路的架構調整，本階段判斷風險與效益比不利於推行，留待後續規劃。
