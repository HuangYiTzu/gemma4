# Attention HLS 整理(給同學):模組總覽 + Tile 處理方式

> 範圍:Gemma-4 E2B text、**decode**(每步 1 個 token,Br = 1)、8 個 Q head 共用 1 組 K/V(MQA)、目標板 U280。
> 權威順序:`golden/attention_golden.py` > `spec/frozen_spec_v1.md` > 本文件。本文件是閱讀用摘要,細節以 golden 與 spec 為準。
> 結構:**Part A** 目前完成了什麼(模組、資料流、驗證、綜合估計、待辦);**Part B** tile 要怎麼處理(接 attention 核心必讀)。

---

# Part A:目前 HLS 完成了什麼

## A1. 資料流與模組分工

```
x_in(int8) → [q/k/v 投影] → [norm(外部)] → [RoPE] → [kv_quant: INT16→int8] → [KV cache]
                                                 └→ Q(INT16) ─────────────────────────→ [attn_core: QK^T → online softmax → PV → o_in] → [o_proj]
```

| 模組 | 路徑 | 作用 | 狀態 |
|---|---|---|---|
| RoPE | `hls/rope/` | 整數 RoPE(split-half),只讀 golden 匯出的 cos/sin 表 | ✅ C 模擬逐位元一致 |
| exp2 / 1/ℓ 查表 | `hls/softmax_aux/` | softmax 用的 8 段分段線性 exp2 與 256 項 1/ℓ 查表 | ✅ C 模擬逐位元一致 |
| KV cache | `hls/kv_cache/` | int8 的 K/V 儲存、依 bank 切分,sliding 環形緩衝 / global 線性 | ✅ C 模擬通過 |
| kv_quant | `hls/kv_quant/` | K/V 寫入 cache 前 INT16 → int8 量化 | ✅ C 模擬逐位元一致(**乘數位寬與 spec 暫定值不同,見 A5**)|
| attn_core | `hls/attn_core/` | tile 迴圈:QK^T、online softmax、PV、收尾量化成 o_in(int8)| ✅ C 模擬逐位元一致 |
| q/k/v/o 投影 | — | 預計重用 MLP 同學的矩陣乘法引擎 | ⛔ 尚未開始(golden 也沒有投影的 `M[c]`)|
| norm | — | 其他同學負責;介面見 `spec/norm_attention_interface.md` | 🟡 格式等對方確認 |

每個模組資料夾的結構一致:`<模組>.h`(核心,template 參數化)、`<模組>_top.cpp/.h`(綜合用 top,暫定)、`tb_<模組>.cpp`(C 模擬 testbench)。

## A2. 各模組重點

**RoPE(`hls/rope/rope.h`)**
- `y1 = (x1·cos − x2·sin + 2^11) >> 12`,`y2 = (x2·cos + x1·sin + 2^11) >> 12`,兩項相加後才捨入一次,飽和 INT16。
- 角度用**絕對位置** `p`(0..4095)。cos/sin 是 14-bit 表(1.0 = 4096),**只讀表、不用 cosf/sinf**。
- Q 輸入輸出都是小數 10 位、K 都是小數 14 位(🟡 隨 norm 輸出格式)。
- sliding:128 組全旋轉;global:只有 64 組旋轉,其餘 192 組精確旁路。
- 四個 top:`rope_{sliding,global}_{q,k}`(Q 是 8 個 head、K 是 1 個 head)。

**exp2 / 1/ℓ(`hls/softmax_aux/`)**
- `exp2.h`:`t = round_half_up(d·1477 / 2^8)`,取整數部分 `u`(floor)與小數部分 → 8 段分段線性查表 → UQ1.16。
- `recip.h`:ℓ 最高位「之後」的 8 位元當索引,表項為中點 `1/(1+(idx+0.5)/256)`;輸出 `(R, w)`。
- 表 `aux_tables.h` 是**產生檔,不要手改**,由 `golden/export_softmax_aux_vectors.py` 產生。

**KV cache(`hls/kv_cache/kv_cache.h`)**
- 一個 word = `LANES` 個 int8,`NB` 個 bank 一次平行讀 `NB` 個位置。`bank = a mod NB`、`row = a / NB`。
- 同類型的 producer 層共用一塊記憶體,`layer` 當高位位址:sliding 12 層(layer 0,1,2,3,5,6,7,8,10,11,12,13 → 索引 0..11)、global 3 層(layer 4,9,14 → 索引 0..2)。consumer 層不需要自己的 cache(sliding 讀 layer 13、global 讀 layer 14)。
- 容量:sliding 12×256KB = 3MB、global 3×4MB = 12MB,共 15MB。
- 記憶體型態(URAM/BRAM)**尚未綁定**。

**kv_quant(`hls/kv_quant/kv_quant.h`)**
- `code = clip(round_half_up(x·M / 2^B), −127, 127)`,每層、K/V 各一組 `(M, B)`,乘積保留完整位寬後才加 `2^(B−1)` 並算術右移。
- 常數表 `kvq_consts.h` 是產生檔(`golden/export_kv_quant_vectors.py`),輸出接 `KvCache::write_row`。

**attn_core(`hls/attn_core/attn_core.h`)**
- 兩個綜合用 top:`attn_core_sliding`(D=256、cache 512、window 512、12 層)、`attn_core_global`(D=512、cache 4096、純 causal、3 層)。
- 輸入:`q[8][D]`(INT16)、K/V cache 的 bank 陣列、`layer`、`p`、`m_s`、`c2`;輸出:`code[8][D]`(int8 的 o_in)。
- template 參數 `<D, CACHE_LEN, WINDOW, NLAYERS, H, LANES, NB, EXP_PAR, HEAD_PAR>`;目前綜合用預設 `HEAD_PAR=8, LANES=8, NB=1, EXP_PAR=8`(🟡 暫定)。
- **`HEAD_PAR`**:Gemma 是 MQA,一次 K/V 讀取同時餵給多個 head 的乘法器;結果與 `HEAD_PAR=1` 完全相同。
- tile 細節見 Part B。

## A3. 驗證方式(C 模擬,逐位元)

所有模組都用同一套流程:`golden/export_*.py` 匯出向量 → `hls/<模組>/tb_*.cpp` 讀向量逐位元比對。

| 模組 | 向量產生指令(專案根目錄)| 驗證內容 |
|---|---|---|
| RoPE | `python golden/export_rope_vectors.py` | 14 個案例,對 golden `rope_fixed` |
| exp2 / 1/ℓ | `python golden/export_softmax_aux_vectors.py` | exp2 對 d = −32767..0 **窮舉**;recip 約 21 萬筆結構化 + 隨機 |
| KV cache | `python golden/export_kv_cache_vectors.py` | 對獨立的 Python 環形緩衝模型;多組 (LANES, NB) |
| kv_quant | `python golden/export_kv_quant_vectors.py` | 30 組 (層, K/V) × INT16 全部 65536 個輸入**窮舉**;另有 quant → write_row → 讀回的端到端測試 |
| attn_core | `python golden/export_attn_core_vectors.py` | 8 個案例(真實層 0/4/13/14、合成、環形緩衝繞回、T=4096);逐 tile 的 m、ℓ、O 與獨立純整數參考比對;最終 o_in 對 golden 預填第 p 列;多組並行度結果必須完全相同 |

- 匯出腳本會先用獨立的純整數公式重算,與 golden 比對後才寫檔,不一致就中止。
- attn_core 的向量資料夾約 42 MB(約 49 個檔),**不建議放進 git**,請用腳本產生。
- 執行 C 模擬:在 Vivado HLS 建專案,把 `<模組>.h`、`*_top.cpp/.h` 設為 design files、`tb_*.cpp` 設為 testbench,top function 選對應的 top;testbench 的第一個參數是向量資料夾(預設 `vectors`)。attn_core 的 testbench 另有 `quick` 參數,只跑綜合用的預設組態(Vivado csim 是 debug 模式,跑完整組合會很慢)。
- golden 本身的測試:`python golden/tb_attention.py --data data --cfg golden/attention_layer_config.csv`(21 項)。

## A4. 綜合估計(Vivado HLS C synthesis,**不是 U280 的數字**)

> 條件:Vivado HLS 2019.2、part 為 `xc7vx485tffg1157-1`(Virtex-7;U280 在這個版本無法選)、時脈 4 ns。以下是 HLS 估計值,不是實作後的結果。

| 模組 | 結果 |
|---|---|
| RoPE | 四個 top 都 II=1,4 DSP,約 650~720 LUT;latency:sliding_q 1236、sliding_k 268、global_k 332、global_q 2196 週期 |
| exp2 | II=1,2 DSP,迭代延遲 13 週期 |
| 1/ℓ | II=1,0 DSP,1 BRAM18,延遲 9 週期 |
| attn_core(`HEAD_PAR=8, LANES=8, NB=1, EXP_PAR=8`) | 所有主要迴圈 II=1;`tile_update` sliding 約 4239、global 約 8400 週期;約 448 DSP、約 120k LUT(40%)、BRAM 260~324;時序估計 3.5 ns + 0.5 ns 不確定 = **剛好 4 ns,幾乎沒有餘裕** |
| attn_core 最壞情況 token(估計) | sliding 約 38.6k 週期(0.15 ms)、global 約 538k 週期(2.15 ms);35 層(28 sliding + 7 global)在最長 context 約 19 ms |

kv_quant 尚未綜合。

**綜合過程中學到的事:**
- 對 `HEAD_PAR` 個 head 的小迴圈**一定要 UNROLL**,否則 HLS 會依序執行,每個 tile 多花約 1136 週期。
- Q 放在雙埠 RAM 時 `qk_loop` 會變 II=4;把 Q 先載入成完全分割的 `[CHUNKS][LANES]` 陣列後修好。
- 時脈放寬到 7 ns 反而因為 HLS 把乘法與加法串在一起,估計成 8.36 ns 而失敗,所以時脈維持 4 ns。
- 用 `LANES=1`、單埠 cache 時速度被記憶體埠限制;要加速必須用寬的 / 分 bank 的 cache 讀取。

## A5. 與 spec 不一致或需要確認的地方

| 項目 | 狀況 |
|---|---|
| kv_quant 乘數位寬 | spec §8 暫定 16 bit,但對 30 組 (層, K/V) 窮舉 65536 個輸入,16 bit 每組有 2~18 個輸入差 1 LSB;**24 bit(B = 28..30)才全部一致**。程式用 `MW = 24`。**spec 檔尚未修改**,等負責人決定。|
| norm 輸出格式 | Q/V 小數 10 位、K 小數 14 位,是依 `norm_attention_interface.md` 的需求暫定,等 norm 同學確認。`F_K`、`F_V`、RoPE 的 IN/OUT_FRAC 都是 template 參數。|
| 投影 `M[c]`、K/V 量化乘數 | golden 沒有模型;kv_quant 目前以 golden 既有的浮點 `quantize` 當標準答案。|
| 輸出 `o_in` | 目前 attn_core 的輸出是 int8 的 o_in(`C2` 與 `s_o_in` 折在一起)。|
| golden 的 decode 路徑 | decode 位置 p 的預期輸出取預填 golden 的第 p 列;`export_attn_core_vectors.py` 已用獨立的 Br = 1 純整數參考直接確認兩者相同。|

## A6. 還沒做 / 下一步

- [ ] q/k/v/o 投影(重用 MLP 同學的矩陣乘法引擎;o_proj 歸約長度 2048 或 4096)
- [ ] 把各模組串成完整 attention 層,與 norm 對接
- [ ] 用 U280 對應的 part / 平台重新綜合,檢查 SLR 分配與資源
- [ ] KV cache 綁定 URAM(2019.2 的 pragma 語法尚未確認)、attn_core 內小陣列改用 LUTRAM 以降低 BRAM
- [ ] 時序餘裕:目前 attn_core 估計剛好 4 ns,需決定放寬時脈或降低並行度
- [ ] kv_quant 綜合
- [ ] `HEAD_PAR` 以外的並行度與目標吞吐量(⛔ 取決於 DSP/URAM 預算)
- [ ] global 層 RoPE 目前仍讓 192 組旁路維度經過乘法器,可跳過以加速(位元結果不變)
- [ ] 4096 長度、35 層、多種文字的驗證(golden 只驗證 6 層、T = 1024)

## A7. 環境與注意事項

- **只有 Vivado HLS 2019.2**:前端預設不是 C++11,程式碼維持 **C++98 相容**(不用 `constexpr`、`static_assert`,改用 `ROPE_STATIC_ASSERT` / `AUX_STATIC_ASSERT` 巨集)。
- Vivado HLS 的編輯器會把檔案存成舊編碼導致中文註解亂碼:**請維持 UTF-8**。目前 `hls/rope/rope.h` 與 `rope_top.h` 的註解已有亂碼(不影響編譯),修正時請用 UTF-8 重存,不要用 HLS 內建編輯器。
- Python 需要 numpy。
- 所有右移都是**算術右移**;`round_half_up` = 加 `2^(s−1)` 再右移。
- 凍結規格(位寬、捨入、Bc、tile 邊界、cache 格式)不要自行更改;🟡 / ⛔ 項目請做成參數並註明「暫定」。
- `aux_tables.h`、`kvq_consts.h`、`*/vectors/` 都是腳本產生的檔案,不要手改。

---

# Part B:Tile 處理方式

> 對應程式:`hls/attn_core/attn_core.h`(tile 迴圈)、`hls/kv_cache/kv_cache.h`(位址)。

## 1. 一句話版本

**K/V 以 64 個位置為一個 tile(Bc = 64),tile 邊界固定在「絕對位置」的 64 的倍數,由小到大逐 tile 走訪,每個 tile 做一次 online softmax 更新。**

不照這三件事做(Bc = 64、絕對位置切、由小到大),輸出會與 golden 不一致。

## 2. Tile 的定義

| 項目 | 規則 | 狀態 |
|---|---|---|
| Bc(一個 tile 的 K/V 位置數) | **64** | ✅ 凍結 |
| Br(一次處理幾個 query) | decode = 1。Br 不影響結果(Br=100 與 64 逐位元相同) | ✅ |
| tile `j` 涵蓋的位置 | 絕對位置 `64j .. 64j+63` | ✅ |
| 走訪順序 | j 由小到大,tile 內位置也由小到大 | ✅ |

**為什麼 Bc 必須固定:** online softmax 每個 tile 會重新算 running max 並對 ℓ、O 做 `α` 縮放與捨入。Bc = 32 或 128 時,有 0.3%~0.6% 的輸出 code 會差 1 LSB。

## 3. 最容易搞錯的地方

### 3.1 tile 邊界用「絕對位置」,不隨 window 移動
sliding 層的 window 是 512,但 tile 邊界**不是**從 `p − 511` 開始切,而是永遠切在 `64 的倍數`。
視窗邊界那一塊 tile 只有一部分位置有效,**用遮罩處理**。

### 3.2 RoPE 用絕對位置,cache 實體位址才用 `pos mod 512`
- RoPE 角度:`p = 絕對位置`(0..4095)。**不要用 `p mod 512`**(會讓位置 > 512 之後的輸出誤差高達 46%~60%)。
- sliding 層 KV cache 是 512 筆環形緩衝,實體位址 `a = pos mod 512`。
- 因為 64 整除 512,**每個 tile 在環形緩衝中是一段不繞回的連續位址**,所以不用處理 tile 內繞回。

### 3.3 被遮罩的位置,P̃ 必須由邏輯強制為 0
不能靠 exp2 下溢得到 0。遮罩條件:

```
valid(pos) = (pos <= p) && ( global 層 ? true : (p - pos) < 512 )
```

- `pos > p`(尚未寫入的位置)視同被遮罩。
- 遮罩處:`d = 0`(不參與 max)、`P̃ = 0`(強制)。
- running max `m` 的**初值是 −2^14**(S 的最小值),**不是 −inf**。

## 4. 哪些 tile 要處理、哪些跳過

對目前 token 位置 `p`:

```
j_hi = p / 64                                  // 最後一個 tile(含 p 的那塊)
j_lo = 0                                       // global 層
j_lo = (p - 512 + 1 > 0) ? (p - 511) / 64 : 0  // sliding 層
處理 j = j_lo .. j_hi(由小到大)
```

- 整塊在未來(`j > p/64`)→ 不處理。
- sliding 層整塊在 window 外(`p − (64j+63) ≥ 512`)→ **跳過**(整塊被遮罩的 tile 不會改變 m、ℓ、O 的狀態,所以跳過與處理結果相同)。
- 邊界 tile(第一塊與最後一塊)只有部分位置有效 → 用遮罩。

範例(sliding,p = 700):`j_hi = 10`、`j_lo = (700−511)/64 = 2`,處理 tile 2..10,共 9 個。
tile 2 涵蓋位置 128..191,其中 `p − pos < 512` 即 `pos > 188` 才有效,所以只有 189..191 有效,其餘遮罩。

## 5. 每個 tile 內做什麼(每個 head、decode)

```
初始(每個 head):m = −2^14,ℓ = 0,O[D] = 0

對 tile j(由小到大):
 1. S_acc[c] = Σ_d Q[d] · K[c][d]            Q: INT16(1+5+10),K: int8,累加 32-bit 有號
 2. S[c] = clip( round_half_up(S_acc·M_S / 2^24), −2^14, 2^14−1 )
 3. m_new = max(m, max_{valid c} S[c])
 4. P̃[c] = valid ? min( round_half_up( exp2(S[c] − m_new) / 2^4 ), 4096 ) : 0    UQ1.12(1.0 = 4096)
 5. α = exp2(m − m_new)                        UQ1.16
 6. ℓ = round_half_up(α·ℓ / 2^16) + Σ_c P̃[c]                  小數 12 位
 7. O[d] = round_half_up(α·O[d] / 2^16) + Σ_c P̃[c]·V[c][d]    小數 12 位
 8. m = m_new

全部 tile 做完 → 收尾:
 (R, w) = Recip(ℓ)                              查表,索引是 ℓ 最高位「之後」的 8 位元
 y    = round_half_up(O·R / 2^(14+w))           小數 10 位
 code = clip( round_half_up(y·C2 / 2^22), −127, 127 )   → o_in(int8)
```

重點:
- **所有 softmax 內部都是整數**,沒有浮點。
- `round_half_up` = 加 `2^(s−1)` 再做**算術右移**(負數也一樣,剛好 .5 往 +∞)。
- `M_S`、`C2` 是**每層一個常數**(由 `golden/attention_layer_config.csv` 的 scale 算出),執行期參數。

## 6. 位寬(最壞情況、不加飽和)

| 訊號 | 格式 |
|---|---|
| Q | INT16 |
| S_acc | 32-bit 有號 |
| S、m | 15~16-bit 有號(1+6+8,單位 2^−8)|
| P̃ | UQ1.12(13-bit)|
| α | UQ1.16(17-bit)|
| ℓ | 25-bit 無號,小數 12 位 |
| O | 32-bit 有號,小數 12 位 |
| 1/ℓ 表 | 256 項 × 13-bit,表項 `1/(1+(idx+0.5)/256)`(中點)|

## 7. HLS 實作怎麼對應

`attn::AttnCore<D, CACHE_LEN, WINDOW, NLAYERS, H, LANES, NB, EXP_PAR, HEAD_PAR>`(`hls/attn_core/attn_core.h`)

| 函式 | 作用 |
|---|---|
| `tile_range(p, j_lo, j_hi)` | 決定要走哪些 tile(第 4 節)|
| `init_state` | `m = −2^14`、`ℓ = 0`、`O = 0` |
| `tile_update` | 一個 tile 的 1~8 步(QK^T → 最大值 → P̃ → α/ℓ → PV → O) |
| `finalize` | 1/ℓ 查表、y、o_in 量化 |
| `run` | `for head group { init → for j in [j_lo, j_hi] tile_update → finalize }` |

K/V 從 `kv::KvCache`(`hls/kv_cache/kv_cache.h`)的 bank 陣列讀:
- 一個 word = `LANES` 個 int8;`NB` 個 bank 一次讀 `NB` 個位置。
- 一個 tile 的 64 個位置分成 `64 / NB` 個「位置群組」;QK^T 與 PV 各把 (群組, chunk) 攤平成一個 pipelined 迴圈。
- 實體位址:sliding `pos mod 512`,global `pos`(必須 < 4096)。
- `HEAD_PAR`:8 個 Q head 共用同一次 K/V 讀取,各 head 的整數運算互相獨立,結果與 `HEAD_PAR = 1` 完全相同。

## 8. 暫定 / 未決(請做成參數,不要寫死)

| 項目 | 狀態 |
|---|---|
| `LANES`、`NB`、`EXP_PAR`、`HEAD_PAR`(並行度)| ⛔ 吞吐量與 URAM/DSP 預算未決,目前預設值是暫定 |
| `M_S`、`C2` 的埠寬(20、16 bit)| 🟡 暫定(實測 35 層 M_S ≤ 15 bit、C2 ≤ 14 bit)|
| norm 輸出格式(Q、K、V 的 INT16 格式)| 🟡 等 norm 負責人確認 |
| K、V 寫入 int8 cache 的量化乘數 | 🟡 golden 尚未建模 |
| 投影(q/k/v/o)的 `M[c]` | 🟡 golden 尚未建模 |
| decode 專用 golden 路徑 | ⛔ 預期輸出取「預填 golden 的第 p 列」;Br = 1 尚未直接驗證 |
| 逐 tile 中間訊號傾印 | ⛔ 尚未實作 |
| 4096 長度、35 層驗證 | ⛔ 只驗證 6 層、T = 1024 |

## 9. 如果你要接這個模組 / 自己寫 tile 迴圈

檢查清單:

- [ ] Bc = 64,tile 邊界在絕對位置的 64 倍數。
- [ ] tile 由小到大走訪,tile 內位置也由小到大。
- [ ] 遮罩處 P̃ 由邏輯強制為 0。
- [ ] `m` 初值 = −2^14。
- [ ] RoPE 用絕對位置;只有 cache 位址用 `pos mod 512`(sliding)。
- [ ] 每個右移都是算術右移,且 round-half-up 寫明(加 `2^(s−1)` 再移)。
- [ ] 與 golden 逐位元比對(不接受「差不多」)。

## 10. 相關檔案

- 規格:`spec/frozen_spec_v1.md`(第 5、6 節)
- Golden:`golden/attention_golden.py`(`attention_fa2_int`、`exp2_int`、`recip_int`)
- 向量匯出:`golden/export_attn_core_vectors.py`、`golden/export_kv_cache_vectors.py`
- Testbench:`hls/attn_core/tb_attn_core.cpp`、`hls/kv_cache/tb_kv_cache.cpp`
- 層設定:`golden/attention_layer_config.csv`
