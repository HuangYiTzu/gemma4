# Residual累加：方案A vs 方案B量化誤差對照實驗

日期：2026-09-19

---

## 1. 為什麼要先做這個實驗，才決定進不進HLS

`layer_scalar`（見`rmsnorm_golden_model.py` checklist B0-2、`residual_golden_model.py`開頭說明）該在「三個分支全部加完之後才乘一次」（方案A，忠於HF官方float forward的運算順序）還是「每一項各自先乘再加總」（方案B，硬體上可能有不同的資源分配選擇），這個決定**直接決定RMSNorm輸出端的Q-format跟資料流結構**：

- 如果方案A是正解，residual buffer中途需要扛得住「三個分支加總、還沒乘layer_scalar」的未縮放大值（Layer 14最壞可到~660），這正是`rmsnorm_golden_model.py` checklist A5選用`S_RES=2^-5`的依據。
- 如果方案B在精度上一樣好（甚至更好），residual buffer可能可以全程維持「已縮放」的小值，`S_RES`或許不需要犧牲到F=5這麼多。

這不是「原則上要驗證完再進HLS」的通用原則，是因為**這個決定會實際改變常數表格式跟runtime乘法器的資料寬度**，選錯了要重來的成本很高，值得先花時間把它测清楚。

## 2. 前置工作：補齊之前完全沒驗證過的兩個分支

在這次實驗之前，`rmsnorm_golden_model.py`裡的呼叫點只有`pre_feedforward_layernorm`/`post_feedforward_layernorm`（MLP分支）跟真實QAT checkpoint比對過（見`rmsnorm_const_table_redesign.md`）。`post_attention_layernorm`（Attention分支）跟`post_per_layer_input_norm`（PLE分支）雖然golden model程式碼裡有實作`call_post_attention_layernorm()`/`call_post_per_layer_input_norm()`，但從來沒有用真實activation驗證過——residual的完整正確性判斷不能只靠MLP這一條分支，這次一併補上。

### 2.1 確認HuggingFace官方forward()的真實資料流

不能用猜的，直接讀`transformers`套件裡`modeling_gemma4.py`的`Gemma4TextDecoderLayer.forward()`原始碼（1388~1445行），確認了：

```python
residual = hidden_states
hidden_states = post_attention_layernorm(self_attn(input_layernorm(hidden_states)))
hidden_states = residual + hidden_states          # Add①：Attention分支

residual = hidden_states                           # residual更新成post-Attention值
hidden_states = post_feedforward_layernorm(mlp(pre_feedforward_layernorm(hidden_states)))
hidden_states = residual + hidden_states          # Add②：MLP分支（用①之後的residual）

if hidden_size_per_layer_input:
    residual = hidden_states                       # residual再更新成post-MLP值
    hidden_states = post_per_layer_input_norm(...)
    hidden_states = residual + hidden_states       # Add③：PLE分支

hidden_states *= self.layer_scalar                 # 全部加完才乘一次！
return hidden_states                                # 變成下一層的residual
```

兩個關鍵確認：
1. **residual累加是序列式的**，不是三個分支各自獨立加回同一個原始residual（`rmsnorm_golden_model.py`的`__main__`示範程式碼裡曾經寫錯，三個分支共用同一個`residual`變數——那只是示範每個`call_xxx()`能不能跑，不是正確的資料流順序）。
2. `layer_scalar`確實是在**全部三個分支加完之後才乘一次**，而且實測全35層的值範圍0.027~0.89（見`residual_golden_model.py`的`LAYER_SCALAR`表），全部小於1、不是恆等於1的佔位buffer——這很可能是LayerScale類的深層transformer穩定化技巧，用來抑制residual stream隨層數累積無界增長。

### 2.2 抓真實activation（`extract_real_residual_activation.py`）

比照`extract_real_rmsnorm_activation_batch.py`的hook手法，這次要多抓5個新的點（原本只有MLP分支的`residual_after_add1`/`down_proj_out`/`mlp_delta`）：

| 抓取點 | 對應HF forward()裡的位置 |
|---|---|
| `o_proj_out` | `post_attention_layernorm`的輸入（Attention的o_proj原始輸出） |
| `attn_delta` | `post_attention_layernorm`的輸出 |
| `residual_after_add2` | `per_layer_input_gate`的輸入（= MLP分支加完後的residual） |
| `per_layer_proj_out` | `post_per_layer_input_norm`的輸入 |
| `ple_delta` | `post_per_layer_input_norm`的輸出 |
| `layer_output` | 整個decoder layer模組本身的輸出（= Add③後再乘layer_scalar，會變成下一層的residual） |

用跟之前MLP分支驗證一樣的6層（0、9、14、15、31、34）、4種prompt，共75個token/層，存成`outputs/real_residual_layer{L}.npz`。

## 3. Attention/PLE分支golden model驗證結果

用跟`rmsnorm_verify_pre_post_mlp.py`一樣的(A)/(B)三方比對結構：

| Layer | Attention (A)浮點vs真實 | Attention (B)整數golden vs真實 | PLE (A)浮點vs真實 | PLE (B)整數golden vs真實 |
|---|---|---|---|---|
| 0 | 1.000000 | 0.999911 | 1.000000 | 0.999994 |
| 9 | 1.000000 | 0.999771 | 1.000000 | 0.999645 |
| 14 | 1.000000 | 0.999890 | 1.000000 | 0.999931 |
| 15 | 1.000000 | 0.999868 | 1.000000 | 0.999996 |
| 31 | 1.000000 | 0.999563 | 1.000000 | 0.997591 |
| 34 | 1.000000 | 0.999692 | 1.000000 | 0.999932 |

**(A) 全部6層都是精確的1.000000**——代表Attention/PLE分支的RMSNorm演算法理解（gamma、eps、mean(x²)維度）完全正確，這是`call_post_attention_layernorm()`/`call_post_per_layer_input_norm()`第一次被真實checkpoint驗證，結果乾淨通過。

**(B) 整數golden model**全部在0.9976~0.9999之間，量級跟已經驗證過的MLP分支一致，沒有發現新的架構性問題（例如像Layer 0 MLP那樣的Step0溢位）。這兩個分支的golden model**可信賴**，可以放心拿來當Plan A/B實驗的輸入。

## 4. Plan A vs Plan B 對照實驗

### 4.1 實驗設計裡的一個數學修正（寫第一版時的疏漏）

第一版`residual_golden_model.py`裡的`residual_add_with_layer_scalar_folded()`只把`layer_scalar`乘到delta上，沒有乘到`residual_in`本身——這其實是不完整的方案B，重新展開：

```
layer_scalar * (residual_in + attn_delta + mlp_delta + ple_delta)
  = layer_scalar*residual_in + layer_scalar*attn_delta
  + layer_scalar*mlp_delta   + layer_scalar*ple_delta
```

**四項都要乘`layer_scalar`**才跟方案A數學上等價，不是只乘三個分支delta——因為官方forward()裡`residual = hidden_states`只是變數重新賦值，最後`hidden_states *= layer_scalar`是對整條累加鏈（包含最原始的`residual_in`）一次乘完，不是只作用在新算出來的delta上。已經在`residual_golden_model.py`裡把這個函式改名重寫成`run_decoder_layer_residual_planB()`，四項（`residual_in`+3個delta）各自呼叫`scale_and_round()`獨立捨入，再加總。

### 4.2 兩個方案的實際運算

```python
# 方案A（run_decoder_layer_residual）：
sum_int = residual_in + attn_delta + mlp_delta + ple_delta   # 整數加法，無精度損失
final   = round(sum_int * layer_scalar)                       # 只捨入一次

# 方案B（run_decoder_layer_residual_planB）：
r = round(residual_in * layer_scalar)   # 獨立捨入
a = round(attn_delta  * layer_scalar)   # 獨立捨入
m = round(mlp_delta   * layer_scalar)   # 獨立捨入
p = round(ple_delta   * layer_scalar)   # 獨立捨入
final = r + a + m + p                    # 整數加法（此步驟本身無損）
```

無限精度下兩者結果相同；定點下差別在於方案A只有1次捨入、方案B有4次獨立捨入——這正是題目要驗證的地方：4次獨立捨入的誤差會不會因為統計上大致抵消而跟1次捨入差不多，還是會顯著疊加變差。

### 4.3 實驗結果：方案A全面勝出

用第3節驗證過的Attention/MLP/PLE三個分支的golden model輸出當delta，`residual_in`則用真實浮點值反推（`residual_in = residual_after_add1_真實 - attn_delta_真實`，這一步是精確減法沒有量化誤差，只是用來還原「進入這一層之前」的真實residual狀態），跟真實`layer_output`（decoder layer模組本身的真實輸出）比對：

| Layer | 方案A cos_sim (min/mean/max) | 方案A MAE | 方案B cos_sim (min/mean/max) | 方案B MAE | 誰的MSE較小 |
|---|---|---|---|---|---|
| 0 | 0.999736/0.999810/0.999908 | 0.0079 | 0.999292/0.999490/0.999760 | 0.0121 | **A** |
| 9 | 0.992132/0.995617/0.996914 | 0.1157 | 0.991987/0.995571/0.996853 | 0.1166 | **A** |
| 14 | 0.993059/0.995551/0.997485 | 0.1164 | 0.993007/0.995507/0.997426 | 0.1170 | **A** |
| 15 | 0.997731/0.998410/0.998999 | 0.0825 | 0.997699/0.998378/0.998976 | 0.0832 | **A** |
| 31 | 0.999820/0.999927/0.999966 | 0.0197 | 0.999819/0.999906/0.999955 | 0.0222 | **A** |
| 34 | 0.999897/0.999962/0.999980 | 0.0095 | 0.999706/0.999897/0.999948 | 0.0156 | **A** |

**全部6層、沒有例外，方案A的MSE都比方案B小。** 彙整全部token：

```
方案A 全部token cos_sim: min=0.992132 mean=0.998213 max=0.999980
方案B 全部token cos_sim: min=0.991987 mean=0.998125 max=0.999955
平均cos_sim差距 (A-B) = +0.0000879
```

平均差距看起來不大（0.0000879），但**方向完全一致**（6/6層都是A贏），不是雜訊；個別層的MAE相對差距更明顯，例如Layer 0（0.0079 vs 0.0121，方案B誤差多53%）、Layer 34（0.0095 vs 0.0156，方案B誤差多64%）。

## 5. 結論與決定

**採用方案A**（忠於HF官方forward順序：先加總、最後才乘layer_scalar捨入一次），理由：
1. 這次實測（不是理論猜測）確認方案A在全部抽樣層的量化誤差都比方案B小。
2. 方案A的實作也更簡單（`rmsnorm_golden_model.py`跟`residual_golden_model.py`目前就是照這個順序寫的，不需要額外改動）。
3. 方案B唯一的潛在優勢（如果residual buffer全程只需要處理「已縮放」小值，或許能縮小`S_RES`位寬）並沒有實現——因為方案B裡`residual_in`也要乘`layer_scalar`才數學正確，這代表**方案B並不會讓residual buffer的動態範圍縮小**（`residual_in`乘完`layer_scalar`後確實變小了，但delta是在「加進residual之前」才各自縮放，沒有改變「累加過程中最大可能出現的中繼值」這件事的本質——中繼的部分和(`r`, `r+a`, `r+a+m`)在方案B裡仍然需要跟方案A一樣寬的位元來表示，因為每一步的加法都是在「已縮放」的獨立分量上做，但分量本身（例如`round(attn_delta*layer_scalar)`）的動態範圍跟delta本身的動態範圍是同量級的，只是多乘了一個<1的係數，不足以把所需位元數砍到能用更窄的Q-format）。

**因此`S_RES=2^-5`（F=5）的決定維持不變**（見`rmsnorm_const_table_redesign.md`第15節），`generate_rmsnorm_const_tables.py`不需要因為這次實驗而修改常數表格式——這次實驗確認的是「residual加法的運算順序該怎麼做」，不是「S_RES的Q-format該怎麼選」，兩者是獨立的問題，這次實驗的結果剛好也支持繼續用F=5（因為方案B並沒有提供縮小位元寬度的空間）。

## 6. 這次實驗附帶驗證/發現的東西

- `call_post_attention_layernorm()`跟`call_post_per_layer_input_norm()`**第一次**被真實checkpoint驗證，浮點演算法理解100%正確（cos_sim=1.000000），整數golden model誤差量級跟MLP分支一致，沒有發現新的架構性問題。
- 確認了HuggingFace官方forward()裡residual累加是**序列式**（MLP分支讀的是Attention分支加完後的residual，不是原始residual），修正了`rmsnorm_golden_model.py` `__main__`示範程式碼裡簡化過頭、順序不正確的寫法（該處僅是示範call_xxx()介面，不影響任何已驗證的正確性結論，但容易誤導之後的人以為那是正確資料流，值得注意）。
- 修正了`residual_golden_model.py`第一版方案B實作的數學疏漏（`residual_in`也要乘`layer_scalar`，不是只有delta）。

## 7. 還沒做的事（下一步open items）

- 這次的residual驗證用的是75個token（4個prompt）的小樣本，跟`rmsnorm_const_table_redesign.md`第15節「S_RES用WikiText 20480個token重新校準」的規模不同——如果要對S_RES的F=5做最終定案，理論上residual層級的驗證也應該擴大到WikiText規模，目前只做了小樣本的方案A/B對照，方向性結論（A優於B）應該穩定，但確切的誤差數字還沒有大樣本驗證。
- `layer_scalar`目前在`residual_golden_model.py`裡硬編碼成`LAYER_SCALAR`表，還沒改成跟`read_layer_constants()`一樣直接從safetensors讀。
- 目前只驗證了單一層的residual組裝（用真實浮點反推`residual_in`，不是真的跨層串接35層跑一次完整sequence）；真正的end-to-end驗證應該是接續`rmsnorm_verify_perplexity.py`的做法，把residual golden model也掛進forward hook，跑一次完整35層的perplexity/hidden state比對——但這一步應該要等`rmsnorm_const_table_redesign.md`第16.2節提到的perplexity異常結果（量化後perplexity反而比baseline好，尚未查清楚原因）先釐清，避免把兩個還沒完全理解的實驗疊在一起，混淆歸因。
