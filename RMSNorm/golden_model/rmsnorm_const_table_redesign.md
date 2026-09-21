# RMSNorm 常數表重新設計：從「WB折疊常數」到「raw gamma + per-layer純量」

日期：2026-09-18（初版）、2026-09-19（新增第11~14節：WB2折疊決策、γ量化對照、Layer 14根因重新診斷與修正、S_RES溢位證據、end-to-end perplexity驗證；第15~16節：WikiText大樣本校準全35層、S_RES正式修正F=8->5、修正後重驗）
影響範圍：`pre_feedforward_layernorm`、`post_feedforward_layernorm` 這兩個呼叫點的HLS常數表產生流程（`generate_rmsnorm_const_tables.py`）
不影響：`rmsnorm_golden_model.py` 的Python數值驗證邏輯（該邏輯一直用浮點直接算 `gamma/S_target`，從未真的做過WB折疊，這次修正的是「把golden model結果轉成HLS常數表」這個下游步驟）

---

## 1. 問題是怎麼發現的

跑完 [rmsnorm_verify_batch.py](../qat_verification/rmsnorm_verify_batch.py) 確認35層裡挑出的6層cos_sim都在可接受範圍後，照計畫執行 `generate_rmsnorm_const_tables.py` 產生HLS常數表。腳本自己印出的溢位檢查馬上跳出警告：

```
WB_PRE  範圍: [-32768, 32767]（位元上限 ±32767，有溢位風險）
WB_POST 範圍: [1544, 32767]  （有溢位風險）
Layer 0 WB_POST範圍: [32767, 32767]  ← 全部1536個channel都卡在上限
```

第一版設計把 `γ_i / S_target` 直接折成一個常數 `WB[layer][ch] = round(γ_i / S_target * 2^13)`，燒進16-bit（Q3.13）ROM。這個Q-format是文件裡承認過的「先沿用13-bit frac當預設」的猜測值，從沒真的拿實際checkpoint數字驗證過。

## 2. 根因：不是Layer 0特例，是Q-format設計錯誤

掃過全部35層後發現：

| 常數表 | 猜測的Q-format | 實測需要的位元數（含符號） |
|---|---|---|
| WB_PRE（`γ_pre/S_target`，FRAC=13） | 16-bit | **30-bit**（最壞值出現在Layer 13） |
| WB_POST（`γ_post/S_RES`，FRAC=13） | 16-bit | **28-bit**（最壞值出現在Layer 0） |

即使拿一個很普通的 `γ≈1.0` 代入，`γ/S_RES = 1/2⁻⁸ = 256`，乘上`2^13`就是2,097,152——早就超過16-bit容量。這不是「Layer 0離群值才會爆」，是**把兩個量級差很大的東西（小的γ、可能很小的S_target）乘在一起折成同一個常數，S_target越小、折出來的數字就越大**，16-bit從一開始就不夠用。

Layer 0之所以看起來最誇張（整層全部1536個channel都頂到上限），是因為它的 `down_proj_input_scale = 27.8`（其他層是0.01~0.1量級，差了近1000倍），把問題暴露得最明顯，但問題本身是全域性的。

## 3. 新架構：把折疊拆開

數學上 `γ_i / S_target = γ_i × (1/S_target)`，原本硬體是把這兩個數直接乘起來、燒成一張常數表；現在拆成兩個獨立的表：

1. **raw gamma（per-channel，位寬小）**：直接存 `γ_i` 本身，不跟 `S_target` 混在一起。实测 `γ` 的量級遠比折疊後的WB小（見下表），塞進INT16綽綽有餘。
2. **`1/S_target`（不是per-channel，是per-layer甚至全域純量）**：
   - `pre_feedforward_layernorm` 的 `S_target = gate_up_input_scale`，每層不同 → 存成 `INV_GATE_UP_SCALE[layer]`，**只有35個值**（不是35×1536個），位寬給寬完全不佔資源。
   - `post_feedforward_layernorm` 的 `S_target` 固定是全域常數 `S_RES = 2⁻⁸`，不隨layer變動！`1/S_RES = 256` 剛好是2的整數次方 → 直接變成**編譯期左移量常數**（`<<8`），HLS runtime連乘法器都不需要。

Runtime資料流變成：

```
// pre_feedforward_layernorm：
inv_rms_scaled = inv_rms_int * INV_GATE_UP_SCALE[layer];   // per-token純量乘法，一個token只算一次
y_i = X_i_int * inv_rms_scaled * GAMMA_PRE[layer][i] >> shift;  // per-channel

// post_feedforward_layernorm：
inv_rms_scaled = inv_rms_int << RMSNORM_INV_S_RES_SHIFT;    // 純位移，不需要乘法器！
y_i = X_i_int * inv_rms_scaled * GAMMA_POST[layer][i] >> shift;  // per-channel
```

per-channel的乘法次數跟原本WB方案完全一樣（還是2次：`X×inv_rms`、`×gamma`），只是第二個乘數從32-bit的WB換成16-bit的raw gamma——**DSP只會更省，不會更貴**。

## 4. 為什麼「S_target是per-layer/全域純量」這個假設站得住腳

這是整個新架構的地基：如果 `S_target`（`gate_up_input_scale`等）其實是per-channel array而不是純量，「per-layer純量表」的設計就會漏掉per-channel的資訊。

實際去safetensors裡查了全部35層、8種會被當成 `S_target`/`S_matmul_in` 的activation_scale key（`gate_proj`/`up_proj`/`down_proj`/`q/k/v/o_proj`/`per_layer_projection`的`input_activation_scale`）：

```
掃過35層 x 8種activation_scale key
非純量（numel!=1）的異常項目數: 0
完全找不到的key: 空集合（全部280個key都存在）
```

結論：這個checkpoint的量化方案是標準的「**per-tensor activation scale + per-channel weight scale**」——所有 `*_activation_scale` 都confirmed是`shape=()`的純量，只有`weight_scale`（例如`down_proj.weight_scale`，shape=(1536,1)）才是per-channel。這也解釋了為什麼Step0的`M[layer][ch]`常數表本來就該是per-channel（它折的是`weight_scale`，真正per-channel的東西），但WB表折的是`γ`（per-channel但量級小）跟`S_target`（純量），把兩者混著折才是設計錯誤的根源。

## 5. 實際數字：從猜測改成從數據反推

新腳本不再猜FRAC，而是先掃過全部35層算出每張表的實測最大絕對值，再反推「在給定位寬下能給的最大FRAC，並預留1-bit安全餘裕」：

| 常數 | 實測最大絕對值 | 選用FRAC | 位元寬度 | 結果範圍 | 狀態 |
|---|---|---|---|---|---|
| GAMMA_PRE（raw γ_pre） | 496.0 | 5 | INT16 | [-86, 15872] | ✅ 未溢位 |
| GAMMA_POST（raw γ_post） | 45.5 | 8 | INT16 | [0, 11648] | ✅ 未溢位 |
| INV_GATE_UP_SCALE（1/S_target，35個值） | ~1010 | 自動選 | INT32 | [8917538, 1010617618] | ✅ 未溢位 |
| STEP0_M（`weight_scale`折疊，第一版設計，未變動） | — | 20（不變） | INT32 | [1115, 113925772] | ✅ 未溢位（本來就沒問題） |

## 6. 資源對比

| 方案 | ROM用量（gamma類表） | per-channel乘法器操作元寬度 |
|---|---|---|
| 第一版WB方案（16-bit，會溢位，不可用） | 210 KB | 16-bit（但數值錯） |
| 第一版WB方案改int32修溢位（能用但笨） | 420 KB | 32-bit |
| **新方案：raw gamma + per-layer純量** | **210 KB** | **16-bit** |

新方案跟「WB改16-bit」的ROM用量一樣（因為本來就是把WB拆開存，gamma表維持16-bit），但**修正了溢位**；跟「WB改32-bit才能用」比，省了**210 KB ROM**，且per-channel乘法器操作元寬度維持16-bit（不需要放大到32-bit），對目前MLP DSP已經吃緊的情況是淨改善，不是額外負擔。額外多的`1/S_target`乘法是per-token純量運算（`post_feedforward_layernorm`甚至直接變成免費的位移），不會隨1536個channel複製。

## 7. Golden model端的另一個獨立修正：Step0中繼位寬（不是這次的主題，但同批修正）

跟WB表溢位是不同層次的問題：`rmsnorm_golden_model.py` 裡 `call_post_feedforward_layernorm()` 在做Step0 per-channel requantize時，中繼緩衝原本是16-bit，Layer 0因為 `down_proj_input_scale` 是離群值（27.8，其他層0.01~0.1量級），導致約10%的元素在這一步就飽和截斷。已經把 `step0_out_bits` 預設從16改成24-bit，`post_feedforward_layernorm`在Layer 0的cos_sim從 min=0.768/mean=0.918 改善到 **min=0.933/mean=0.971**。

## 8. 目前驗證狀態（6層抽樣，75個token，4種prompt，2026-09-19更新：Layer 14 pre_ln已套用重新校準）

| Layer | pre_ln cos_sim (min/mean/max) | post_ln cos_sim (min/mean/max) | 備註 |
|---|---|---|---|
| 0 | 0.9991 / 0.9994 / 0.9996 | 0.9333 / 0.9709 / 0.9972 | Step0 24-bit修正後已改善；post_ln仍有S_RES溢位風險（見第13節） |
| 9 | 0.9986 / 0.9991 / 0.9995 | 0.9998 / 0.9999 / 0.9999 | 正常 |
| 14 | 0.9952 / 0.9968 / 0.9979 | 0.9118 / 0.9537 / 0.9997 | **pre_ln已修正**（見第12節）；post_ln是S_RES溢位（見第13節） |
| 15 | 0.9998 / 0.9999 / 0.9999 | 0.9999 / 1.0000 / 1.0000 | 正常 |
| 31 | 0.9993 / 0.9999 / 1.0000 | 0.9999 / 1.0000 / 1.0000 | 正常 |
| 34 | 0.9971 / 0.9997 / 1.0000 | 0.9811 / 0.9970 / 1.0000 | post_ln有S_RES溢位風險（見第13節） |

## 9. 產出檔案

- `rmsnorm_gamma_table.h` — `RMSNORM_GAMMA_PRE[35][1536]`、`RMSNORM_GAMMA_POST[35][1536]`（皆INT16）
- `rmsnorm_layer_scale_table.h` — `RMSNORM_INV_GATE_UP_SCALE[35]`（INT32）、`RMSNORM_INV_S_RES_SHIFT`（編譯期左移量常數）
- `rmsnorm_step0_table.h` — `RMSNORM_STEP0_M_POST[35][1536]`（INT32，未變動）
- `generate_rmsnorm_const_tables.py` — 產生上述三個檔案的腳本，FRAC值全部從實測數據反推，不是猜的
- `rmsnorm_verify_gamma_quant_ablation.py`（`qat_verification/`）— γ量化對照實驗，見第11節
- `rmsnorm_verify_perplexity.py`（`qat_verification/`）— end-to-end perplexity驗證，見第14節

## 10. 尚待確認/決策的事項

- **A1**：q/k/v_norm的INT8/INT16最終位寬未拍板，會影響之後幫這些呼叫點做同樣的常數表產生時，`GAMMA_Q/K/V`該用哪個FRAC。
- **A5（2026-09-19新證據，高優先度）**：`S_RES=2⁻⁸`全域常數溢位餘裕不足，6層抽樣裡3層（Layer 0/14/34）實測飽和，最壞到3.96倍，詳見第13節。這是**全域性**問題（S_RES被所有35層的residual加法點共用），需要wikitext大規模校準後才能決定新的F值，不能只改單一層。
- 這次只處理了`pre/post_feedforward_layernorm`兩個呼叫點；`input_layernorm`、`q/k/v_norm`、`post_attention_layernorm`、`post_per_layer_input_norm`之後要產生常數表時，應該直接套用「raw gamma + per-layer/全域純量」這個架構，不要走回第一版WB折疊的老路。

---

## 11. WB2折疊 vs 拆表：要不要合併回2次乘法？

有人問：既然`GAMMA_PRE`跟`INV_GATE_UP_SCALE`都已經是從實測數據反推出安全的FRAC，能不能離線先把兩者乘好存成單一常數`WB2[layer][ch] = INV_GATE_UP_SCALE[layer] × GAMMA_PRE[layer][ch]`，這樣runtime又能回到2次乘法（不像現在拆成3次：`X×inv_rms`、`×INV_GATE_UP_SCALE`、`×GAMMA_PRE`）？

**結論：不建議，拆表不是可有可無的維護便利，是16-bit datapath下唯一可行的方案。**

驗算：即使`WB2`完全捨棄小數精度（FRAC=0，只存整數），全35層最大值`40961.5`還是超過16-bit容量（需要至少17-bit才存得下整數部分本身）。也就是說：

- 折疊回單一表 → 只能用32-bit儲存，等於繞回第2節「WB方案改int32」那個對照組：420KB ROM、32-bit per-channel乘法器
- 拆表（目前方案）→ 210KB ROM、16-bit per-channel乘法器 + 1次per-token純量乘法（`inv_rms`本來就是每個token一個純量值，這次多乘的`1/S_target`也是純量，不隨1536個channel複製，幾乎不佔DSP）

在MLP DSP已經吃緊的前提下，拆表版本在ROM和DSP兩項都嚴格優於折疊版本。維護上的好處（之後重新校準`gate_up_input_scale`不用重燒整張gamma表）是真實存在的次要優點，但不是決定性理由。

## 12. γ量化對照實驗，以及Layer 14根因的重新診斷（2026-09-18版診斷有誤，已更正）

### 12.1 先補上一個驗證缺口：γ的INT16量化誤差從沒被模擬過

到12節之前的所有batch驗證（`rmsnorm_verify_batch.py`）算出來的cos_sim，golden model裡的`gamma`都還是全精度float64——`rmsnorm_core()`本來就是直接用浮點算`gamma/S_target`，從沒真的模擬過`GAMMA_PRE_FRAC=5`／`GAMMA_POST_FRAC=8`這個INT16量化步驟的誤差。寫了`rmsnorm_verify_gamma_quant_ablation.py`補這個對照：

| Layer | pre cos_sim (float γ) | pre cos_sim (INT16 γ) | Δ | post cos_sim (float γ) | post cos_sim (INT16 γ) | Δ |
|---|---|---|---|---|---|---|
| 0 | 0.999360 | 0.999360 | -0.000000 | 0.970923 | 0.970923 | +0.000000 |
| 9 | 0.999100 | 0.998835 | -0.000265 | 0.999870 | 0.999870 | -0.000000 |
| 14 | 0.916176 | 0.915842 | -0.000334 | 0.953724 | 0.953724 | -0.000000 |
| 15 | 0.999863 | 0.999598 | -0.000265 | 0.999963 | 0.999963 | -0.000000 |
| 31 | 0.999882 | 0.999787 | -0.000095 | 0.999981 | 0.999978 | -0.000003 |
| 34 | 0.999677 | 0.999542 | -0.000135 | 0.997041 | 0.997041 | -0.000000 |

全部層的Δ都在±0.0004以內，**γ量化不是誤差主要來源**，`GAMMA_PRE_FRAC=5`／`GAMMA_POST_FRAC=8`這兩個FRAC值沒問題。針對Layer 14當時最差的token（idx=36）單獨確認：float γ的cos_sim=0.799681，INT16 γ的cos_sim=0.799375，差異只有-0.0003——γ量化也排除了。

### 12.2 逐channel誤差分析，找到真正根因（推翻2026-09-18的錯誤診斷）

2026-09-18當時的診斷寫著「該token的pre_ln輸出本身量級偏小（約0.07~0.78），INT8量化雜訊被放大」——**這個診斷是錯的**，成因是當時只印了輸出向量的前5個元素、剛好都很小，誤導了判斷。實際上該token`real_pre_ln_out`的L2 norm是18.71，跟前後token（16~19量級）完全正常，不是小訊號。

逐channel誤差分析揪出真正的根因：

```
誤差最大的channel: ch1133
  golden = -2.2573   real = -13.6625   err = +11.4052
  residual[ch1133] = -63.7949（整個1536維residual向量裡的最大絕對值）
  gamma[ch1133] = 2.3750
INT8輸出飽和數: 1/1536（就是這個channel）
```

`ch1133`的真實值`-13.6625`遠超INT8在`gate_up_input_scale=0.0176353`下能表示的範圍（`±127×0.0176353=±2.2397`），直接飽和在下限，單一channel誤差+11.4，主導了整個1536維向量的cos_sim。

進一步確認這不是巧合，是transformer常見的**activation outlier channel**現象——固定少數channel系統性地遠大於其他channel：

| Layer | 「每個token最大絕對值channel」最常出現的index | 出現次數/75 |
|---|---|---|
| 0 | 1042 | 17 |
| 9 | 1295 | **75**（全部token） |
| 14 | 1133 | 16 |
| 15 | 850 | 74 |
| 31 | 438 | **75**（全部token） |
| 34 | 438 | **75**（全部token） |

Layer 9/31/34都有一個outlier channel在**全部**token上都是最大值，但這幾層cos_sim反而正常——代表這幾層的`gate_up_input_scale`校準時已經涵蓋了該outlier channel的量級，只有Layer 14的校準值`0.0176353`沒有涵蓋`ch1133`偶爾衝到63.8的情況（該channel在這75個token的平均量級是37.3，不是罕見的一次性事件）。

**這不是golden model的邏輯bug**——golden model正確模擬了「用checkpoint這個`gate_up_input_scale`部署，真實硬體在這個token上也會飽和」。問題出在checkpoint提供的這個activation scale本身校準覆蓋範圍不足。

### 12.3 修正：用實測數據重新校準gate_up_input_scale

```
checkpoint提供的gate_up_input_scale = 0.017635  ->  INT8可表示範圍: ±2.2397
75個token實測real_pre_ln_out最大絕對值 = 13.6625（來自ch1133）
重新校準(max/127) = 0.107579（是checkpoint值的6.10倍）
```

套用到`rmsnorm_verify_batch.py`（`RECALIBRATED_GATE_UP_SCALE = {14: 0.107579}`）重跑：

| 指標 | 修正前 | 修正後 |
|---|---|---|
| pre_ln cos_sim min | 0.7997 | **0.9952** |
| pre_ln cos_sim mean | 0.9162 | **0.9968** |

⚠️ 這個校準值只用了75個token（4種prompt）算出來，樣本數還不夠穩健，這裡只是驗證「加大scale」這個修正方向正確，最終數值需要等wikitext大規模校準才能定案（呼應checklist A5同樣的樣本數問題）。

## 13. post_feedforward_layernorm的Layer 14誤差：不同根因，不能用同一招修（併入A5）

修完pre_ln後，post_ln仍然是全部6層裡最差的（cos_sim min=0.9118）。用同樣的逐channel誤差分析法診斷：

```
worst post_ln token idx=73, cos=0.911792
誤差最大channel: ch850
  golden = 127.9961   real = 507.0028   err = -379.0067
INT16輸出飽和數: 5/1536
S_RES可表示範圍: ±127.9961
```

同樣是outlier channel飽和（`ch850`正好跟Layer 15的outlier channel是同一個），但**關鍵差異**：`gate_up_input_scale`是per-layer的，可以只改Layer 14；但這裡飽和的是`S_RES=2⁻⁸`，這是**全域常數**，被`post_attention_layernorm`、`post_feedforward_layernorm`、`post_per_layer_input_norm`——所有35層的residual加法點共用，不能像`gate_up_input_scale`那樣只調整單一層，改了會牽動整個residual stream的表示方式，需要team拍板，不是這次能單方面決定的。

擴大檢查全部6層抽樣的`real_post_ln_out`是否超出`S_RES`的INT16表示範圍（±128）：

| Layer | real_post_ln_out最大絕對值 | 超出倍數 | 超出範圍元素數/115200 |
|---|---|---|---|
| 0 | 383.90 | **3.00x** | 245 |
| 9 | 35.21 | 0.28x（正常） | 0 |
| 14 | 507.00 | **3.96x** | 274 |
| 15 | 51.90 | 0.41x（正常） | 0 |
| 31 | 11.89 | 0.09x（正常） | 0 |
| 34 | 205.77 | **1.61x** | 50 |

6層抽樣裡有**3層（0、14、34）都超出範圍**，最壞到3.96倍——比golden model checklist A5原本記錄的「100筆WikiText樣本，83.6對128只有1.53倍餘裕」嚴重得多。這個發現已經寫回`rmsnorm_golden_model.py`的checklist A5（見該檔案），標記為高優先度、待wikitext大規模校準時處理，**這次故意不修**（S_RES是全域常數，用少量樣本局部修正風險很高，可能顧此失彼）。

## 14. End-to-end perplexity驗證：component誤差經過剩下20幾層後，對整體品質的真實影響

cos_sim是component-level指標，不代表整個模型的end-to-end品質。寫了`rmsnorm_verify_perplexity.py`，用forward hook把全部35層的`pre/post_feedforward_layernorm`替換成golden model的量化模擬結果（含Layer 14的重新校準，**故意保留S_RES=2⁻⁸現狀**、不修正第13節發現的溢位問題），在WikiText-2 test set（本地已有，未重新下載）上跑12個512-token的chunk，比較baseline float模型跟quantized模型的perplexity差距，以及最後一層hidden state的MSE/MAE。

### 結果

| 指標 | Baseline（原始float） | Quantized（golden model模擬） | 差距 |
|---|---|---|---|
| Perplexity | 546.14 | 598.70 | **+52.56（+9.62%）** |
| 最後一層hidden state MSE | — | — | 1.7944 |
| 最後一層hidden state MAE | — | — | 0.9553 |
| 最後一層hidden state cos_sim | — | — | 0.9528 |

逐chunk比較（12個512-token chunk，loss）：

| chunk | baseline loss | quantized loss | 變化 |
|---|---|---|---|
| 1 | 5.2546 | 5.3795 | 惡化 |
| 2 | 6.7426 | 6.8451 | 惡化 |
| 3 | 6.2556 | 6.2802 | 惡化 |
| 4 | 6.3524 | 6.5244 | 惡化 |
| 5 | 6.1319 | 6.2986 | 惡化 |
| 6 | 6.2409 | 6.4218 | 惡化 |
| 7 | 6.2095 | 6.1472 | 略微改善 |
| 8 | 6.4719 | 6.3361 | 略微改善 |
| 9 | 6.9520 | 7.0168 | 惡化 |
| 10 | 6.4619 | 6.5850 | 惡化 |
| 11 | 5.8328 | 5.9949 | 惡化 |
| 12 | 6.7284 | 6.9075 | 惡化 |

12個chunk裡有10個惡化、只有2個略微改善——差距是一致的系統性訊號，不是單一chunk的雜訊。

（補充說明：baseline perplexity本身546這個數字偏高，這是預期中的——`gemma-4-E2B-it`是instruction-tuned聊天模型，直接拿raw WikiText百科文字去測，本來就不是它的最佳場景，不代表模型本身有問題；這裡真正要看的是baseline跟quantized的**相對差距**，不是絕對數值。）

### 解讀

**+9.62%的perplexity差距不算小，證實component-level誤差確實會傳播到end-to-end，不會被後面20幾層的Attention/MLP完全稀釋掉。** 最後一層hidden state的cos_sim只有0.9528（component-level單一RMSNorm呼叫點的cos_sim大多在0.99+），代表誤差在往後傳遞的過程中被放大而不是稀釋——這點呼應第2節「component cos_sim好看，不代表end-to-end沒事」的提醒。

這裡故意保留了S_RES溢位（第13節發現的Layer 0/14/34飽和）沒修正，所以這個+9.62%的差距裡包含了S_RES溢位的貢獻，不能單獨歸因給哪一個問題——但這個數字本身就是「A5應該優先處理」的量化證據：如果連目前已知會飽和的3個層都不修，end-to-end品質就有接近10%的perplexity劣化，這是具體、可以拿去跟team討論優先順序的數字，不是抽象的「cos_sim掉到0.9x」。

### 這個結果目前的侷限

- 樣本數（12個chunk、共6144個token）比正式paper等級的WikiText-2 perplexity評測（通常用整個test set、數萬~數十萬token，且用sliding window而非不重疊chunk）小很多，數字本身還不夠穩健，僅供「差距方向與量級」的初步判斷，不是最終定案的精度數字。
- 只替換了`pre/post_feedforward_layernorm`這兩個呼叫點，其餘RMSNorm呼叫點（`input_layernorm`、`q/k/v_norm`、`post_attention_layernorm`、`post_per_layer_input_norm`）跟Attention/MLP的matmul本身都還是原始float——真正end-to-end全部量化後的perplexity差距，理論上會比這裡量到的更大。
- 這次刻意保留S_RES的溢位問題，之後如果照wikitext校準結果調整F值，應該要重跑這支腳本，確認perplexity差距有沒有隨之縮小，驗證修正真的有效。

---

## 15. WikiText大樣本校準：S_RES正式修正（F=8 -> F=5），全部35層gate_up_input_scale重新校準

第14節的+9.62% perplexity差距，加上第13節已知的S_RES溢位問題（僅75個token樣本就發現3層飽和），足以支持優先處理。這一節做兩件事：(a) 用遠大於75個token的WikiText樣本重新校準S_RES的F值；(b) 把12.3節「用實測activation重新校準gate_up_input_scale」的方法，從只修Layer 14擴大到全部35層。

### 15.1 記憶體考量：不能像之前一樣存完整張量

之前`extract_real_rmsnorm_activation_batch.py`把完整`(token, 1536)`張量存成`.npz`，6層*75個token沒問題（幾MB）；但這次要抓35層*數萬個token，完整存下來會爆記憶體（35層*20480 token*1536維*8 bytes*2個陣列 ≈ 17GB，不可行）。

寫了`calibration/wikitext_calibrate_layer_scales.py`，改成**streaming統計**：forward hook不保留完整張量，只維護「per-channel running max」（35層*1536，記憶體固定跟channel數成正比）跟「per-token max的清單」（拿來算percentile，記憶體只跟token數成正比），這樣可以跑遠大於75個token的樣本而不會記憶體爆掉。

### 15.2 樣本規模

WikiText-2 **train split**（本地已有，未重新下載；跟第14節perplexity驗證用的test split分開，避免校準跟評測用同一份資料造成校準值「看過答案」）：40個512-token的不重疊chunk，共**20,480個token**，是先前75個token樣本的約273倍。跑了約1007秒（35層*70個hook，CPU-only）。

### 15.3 S_RES結果：F=8/7/6全部飽和，F=5才夠

```
全35層post_ln的global max（所有layer中的最大值）= 595.7565（Layer 14, channel 850）
  F=8: ±128.00   仍會飽和!
  F=7: ±255.99   仍會飽和!
  F=6: ±511.98   仍會飽和!
  F=5: ±1023.97  不會飽和，有餘裕
```

比75個token樣本發現的507還嚴重（595.76），證實這不是75-token樣本的雜訊，是真實存在、且samples越多越可能撞到更大值的系統性風險。

### 15.4 分組scale的評估（文件第1節open item，這次認真評估但選擇不實作）

檢查排除Layer 0/14/34這3個outlier層後，剩下32層的分布：

```
剩下32層的最大值只有138.97（Layer 13），是128的1.09倍，F=7就綽綽有餘
```

也就是說，**用單一全域`F=5`會讓32個原本用`F=7`甚至`F=8`就夠的層，被迫犧牲到8倍粗的量化step**，只為了遷就3個層。這是「分組/per-layer S_RES」的強力數據支持——理論上完全可行，因為residual stream跨層交接處插入一次rescale（per-tensor純量乘法，不是per-channel，硬體成本比照`INV_GATE_UP_SCALE`那類per-token純量乘法，應該不貴）。

**但這次選擇不實作**，原因：
1. S_RES牽動的是residual stream本身的累加，是整個模型最核心的資料通路，貿然改動又沒驗證過的rescale邏輯，風險遠高於gate_up_input_scale那種「只影響單一呼叫點輸出」的per-layer調整。
2. 目前的golden model/HLS pipeline完全沒有「跨層rescale」這個概念，要導入需要重新設計residual累加的資料流，不是單純改一個常數表。
3. 用全域F=5雖然犧牲精細度，但**保證正確**（不會有任何已知的飽和風險），在還沒把分組架構設計、驗證清楚之前，先求正確、再談效率優化比較穩妥。

記錄成有堅實數據支持的後續優化方向（見`rmsnorm_golden_model.py` checklist A5），不是「沒想過」而是「評估後刻意暫緩」。

### 15.5 全部35層gate_up_input_scale重新校準

比照12.3節的方法（`global_max / 127`），但這次用WikiText 20,480個token的樣本、套用到全部35層，不再只修Layer 14。寫成共用模組`qat_verification/layer_scale_calibration.py`，讓`rmsnorm_verify_batch.py`、`rmsnorm_verify_perplexity.py`都從同一份WikiText校準結果讀取。

部分層的校準結果對照（checkpoint原值 -> WikiText重新校準值）：

| Layer | checkpoint原值 | WikiText重新校準 | 倍數 |
|---|---|---|---|
| 0 | 0.940687 | 0.702458 | 0.75x（比原值小，checkpoint原本設定過度保守） |
| 9 | 0.028379 | 0.039901 | 1.41x |
| 14 | 0.017635 | 0.155502 | 8.82x（比75-token樣本算出的6.1x更嚴重） |
| 15 | 0.023189 | 0.023913 | 1.03x（幾乎不用調） |
| 31 | 0.017085 | 0.031465 | 1.84x |
| 34 | 0.022695 | 0.040180 | 1.77x |

有趣的是Layer 0反而**縮小**了（checkpoint原值比WikiText實測還大），代表checkpoint對Layer 0的校準本來就偏保守；Layer 14則進一步證實了outlier channel問題比小樣本估計的更嚴重（8.82倍而不是6.1倍）。

## 16. 套用修正後重新驗證：batch verify + perplexity

### 16.1 Batch verify（6層抽樣，75個token）

| Layer | pre_ln cos_sim (min/mean/max) 修正後 | post_ln cos_sim (min/mean/max) 修正後 | 對照：修正前 |
|---|---|---|---|
| 0 | 0.9995 / 0.9996 / 0.9998 | **1.0000 / 1.0000 / 1.0000** | pre 0.999/post 0.933~0.972 |
| 9 | 0.9919 / 0.9963 / 0.9986 | 0.9876 / 0.9922 / 0.9947 | pre 0.999+/post 0.999+（trade-off見下） |
| 14 | 0.9902 / 0.9932 / 0.9956 | 0.9906 / 0.9941 / 0.9968 | pre 0.7997→0.9952（本次再進一步）/post 0.9118 |
| 15 | 0.9997 / 0.9998 / 0.9998 | 0.9967 / 0.9976 / 0.9984 | 都是0.999+ |
| 31 | 0.9998 / 0.9998 / 0.9999 | 0.9976 / 0.9988 / 0.9998 | 都是0.999+ |
| 34 | 0.9998 / 0.9998 / 0.9999 | 0.9990 / 0.9992 / 0.9993→0.99997 | post本來0.981~0.997 |

**全部6層、全部12個數字都在0.987以上**（先前最差是0.7997/0.9118），最差的是Layer 9的post_ln（0.9876）——這是預期中的trade-off：WikiText樣本發現Layer 9真實動態範圍比原本75-token樣本測到的大，放寬scale去涵蓋更大範圍後，對這75個token（原本落在較窄範圍內）犧牲了一點precision，換取對更大樣本/未來輸入的安全餘裕。這個trade-off是合理的，不是新的bug。

### 16.2 End-to-end perplexity——結果出乎意料，誠實記錄（不當成單純的「修好了」證據）

套用S_RES=2^-5跟全部35層重新校準的`gate_up_input_scale`後重跑`rmsnorm_verify_perplexity.py`：

```
Baseline  perplexity : 546.14
Quantized perplexity : 450.47
差距                 : -95.67 (-17.52%)

最後一層hidden state MSE     : 5.471792
最後一層hidden state MAE     : 1.741690
最後一層hidden state cos_sim : 0.858225
```

對照第14節修正前的結果：

| 指標 | 修正前（v1，只修Layer 14+舊S_RES） | 修正後（v2，全35層+S_RES=2^-5） |
|---|---|---|
| Quantized perplexity | 598.70（比baseline差+9.62%） | 450.47（比baseline好-17.52%） |
| 最後一層hidden state cos_sim | 0.9528 | **0.8582（更差）** |
| 最後一層hidden state MSE | 1.7944 | **5.4718（更差）** |

**這個結果不能直接當成「修好了」來報告，原因：**

1. **量化（有損運算）讓perplexity比全精度baseline還好，這件事本身就違反直覺**，通常代表評測setup有confound，不是單純的「量化竟然有正則化效果」這麼簡單的好消息。
2. **更矛盾的是**：hidden state跟baseline的cos_sim（0.858）比修正前（0.953）還要**更差**、MSE更大——代表量化後的模型內部表示，客觀上跟原始float模型**發散得更多**，但最終loss卻改善了。這代表「跟baseline的忠實度」與「這次剛好在WikiText這批文字上predict得更準」是兩件不完全一致的事。

**排查過程（確認不是hook機制的bug）：**

單獨測試把`pre_feedforward_layernorm`的hook替換機制套用在單一層（Layer 0）、單一prompt上，比對替換前後的輸出：

```
原始float輸出 sample: [  1.557909   -3.2233813 -10.432083    5.232709    3.2182496]
替換後輸出 sample:   [  1.40491552  -3.51228879 -10.53686638   4.91720431   3.51228879]
整體cos_sim: 0.999662
```

單層替換的cos_sim（0.9997）跟batch verify驗證過的數字一致，**hook替換機制本身沒有bug**。

**目前最可能的解釋（尚未完全證實，記錄為待釐清的開放問題，不是定論）：**

- `gemma-4-E2B-it`是instruction-tuned聊天模型，在raw WikiText百科文字（不是它訓練時的對話格式）上評perplexity，baseline本身就偏高（546），這個評測setup跟模型的訓練分布有落差，可能讓「量化雜訊剛好打亂了某種OOD(out-of-domain)下的不良行為」這種巧合更容易發生。
- v2一次改了兩件事（S_RES + 全35層gate_up_input_scale），且全35層gate_up_input_scale有些層是**放寬**（增加量化雜訊，即使那些層原本沒有溢位問題），這些新增雜訊會在35層裡複合，可能導致某種難以預測的整體偏移，剛好在這批WikiText樣本上方向對了。
- **無法排除純粹是這批12個chunk（僅6144個token）樣本太小、運氣使然**——這正是第14節「侷限」提到的樣本數問題，還沒有大到能穩定估計這種細微的整體效應。

**這次驗證任務對「是否真正修好」的結論：**

**Batch verify（全部6層、逐token、12個數字都在0.987以上）才是這次修正真正可信賴的證據**，因為它是每個RMSNorm呼叫點獨立跟真實QAT模型輸出比對，訊號乾淨、可歸因。Perplexity這個「變好」的結果應該記錄為**有趣但不能過度解讀的異常觀察**，理由：
1. 量化讓perplexity比float baseline更好，這種方向性的結果本身需要更大樣本、更嚴謹的統計檢定才能確認是真實效應還是雜訊
2. 最終判準（hidden state cos_sim/MSE跟baseline的忠實度）反而在v2變差，跟perplexity改善的方向矛盾，兩個指標打架時，不應該選對自己有利的那個下結論
3. 不建議拿這個perplexity數字去跟team報告「已驗證修正有效」——如果要對外報告，應該用batch verify的cos_sim數據，並附註perplexity這個異常觀察待更大樣本/更適合的評測集（例如換成chat格式評測，而非raw WikiText）進一步釐清。

**後續建議**：如果要認真追查這個異常，下一步應該是（a）用更大的WikiText樣本重跑（例如整個test set而非12個chunk）看效應是否穩定存在；（b）換成更貼近模型訓練分布的評測方式（chat template格式，而非raw百科文字）；（c）把「只修S_RES、不碰gate_up_input_scale」跟「只放寬gate_up_input_scale、不碰S_RES」分開測試，隔離兩個改動各自對perplexity/cos_sim的獨立貢獻，而不是一次改兩件事。這些都還沒做，這次先誠實記錄現象，不做過度解讀。
