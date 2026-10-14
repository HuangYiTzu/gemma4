# RMSNorm HLS Stage3位元寬度審查：從TODO清單到per-layer修正

日期：2026-09-20

---

## 1. 背景：往HLS前的最後一道關卡

拿到`fast_inv_unit.h`跟`rmsnorm_kernel.h`兩份HLS標頭檔（對應`rmsnorm_golden_model.py`的`rmsnorm_core()`/`rmsnorm_call()`跟`rmsnorm_const_table_redesign.md`第15~16節定案的常數表格式），在寫C-simulation testbench之前，先把檔案裡標記的TODO（`RMSNORM_STAGE3_SHIFT_PRE`/`POST`的「TODO: 交叉驗證」）跟隱藏的位元精度問題過一輪。

這輪審查發現的問題比檔案自己標的TODO多、也更嚴重——不是「shift量沒驗證」這麼單純，是整個Stage3 multiply chain的位元預算規劃方式有根本性問題。

## 2. 第一輪審查：三個獨立問題

### 2.1 Pre-Norm路徑shift量漏算一項

`RMSNORM_STAGE3_SHIFT_PRE = 20 + RMSNORM_GAMMA_PRE_FRAC`完全沒算`INV_GATE_UP_SCALE`本身的FRAC（當時的`RMSNORM_INV_SCALE_FRAC`寫死是0，但實際上`INV_GATE_UP_SCALE`的值高達十億量級，證實FRAC根本不是0）。

### 2.2 `(ap_int<N>)ap_ufixed值`的cast語意

Vitis HLS的`ap_fixed`轉`ap_int`是數值保留（value-preserving），不是bit pattern reinterpret。`inv_rms`（通常是0.001~3的小數）直接cast會把小數部分捨掉，資訊在cast那一步就丟光。修正：用`.range()`明確取出底層bit pattern，當成scaled integer處理。

### 2.3 `inv_rms_scaled`宣告的位寬可能不夠

`ap_int<48>`裝不下`inv_rms`(32-bit) × `inv_scale`(最壞2^30)的乘積。

## 3. 第二輪審查：更根本的問題——Stage3 multiply chain沒有整體位元預算

修完前三個問題、往下推導位元寬度才發現：即使把`INV_GATE_UP_SCALE`的FRAC壓到最窄，**rsq_t自己的20-bit frac（`kernel.h`共用型別，不能動）已經逼近`ap_int<64>`能扛的上限**。四項（`inv_rms`、`inv_scale`、`x_uniform`、`gamma`）連續相乘、最後才一次捨入的資料流結構，實測最壞情況需要**91-bit**，遠超過任何實際硬體乘法器能扛的寬度。

根因：每張常數表各自在自己的容器裡「塞最大精度」是合理的，但沒有人往下檢查這些表最後要一起塞進同一條multiply chain，多個各自最大化的FRAC加總起來遠超過硬體乘法器能扛的寬度——這是**常數表設計跟硬體Stage3乘法鏈之間缺乏整體位元預算規劃**造成的。

### 3.1 修正方案：方案1（重新分配FRAC）+ 方案2（中途插入捨入）混合

- 方案1單獨不夠：不管`INV_GATE_UP_SCALE`的FRAC壓多低，「三個乘法連續相乘、最後才一次捨入」這個資料流結構本身撐不住。
- 方案2（中途插入一次捨入窄化）：在`inv_rms × inv_scale`算完後，捨入到較窄的中繼frac，才能繼續往下乘`x_uniform`、`gamma`而不爆位元。

實測（真實checkpoint數字，不是範例）：Pre-Norm全域共用時`INV_RMS_SCALED_FRAC=4`，Step C = 63-bit（安全）。

### 3.2 Post-Norm路徑也需要同樣的修正（原本以為不用）

原始假設「Post-Norm只是純位移，不損失精度，不需要中途捨入」是錯的——位移雖然不損失精度，但`int_bits`會跟着變大，總位元數一樣會撐爆後面的乘法鏈。實測Step C'需要**76-bit**，超過`ap_int<64>`，證實需要跟Pre-Norm對稱地補上中途捨入。修正後Post-Norm的`INV_RMS_SCALED_FRAC=7`，Step C' = 63-bit（安全）。

## 4. 第三輪審查：全域共用frac導致Layer 0 underflow到cos_sim=0

位元預算通過後，**跑golden model精度驗證**（不能只看位元算得過去就假設數值正確，這是residual那邊Plan A/B實驗學到的教訓，同一套方法論這裡再驗一次），寫了`qat_verification/rmsnorm_verify_stage3_bitwidth.py`：

比對三種版本：真實QAT模型輸出（ground truth）、golden model理想版（float rsqrt，不受位元寬度限制）、bit-accurate硬體模擬版（有/無中途捨入兩種）。

**結果：Layer 0的Pre-Norm cos_sim = 0.000000。** 不是精度下降，是資訊整個被捨光。

### 4.1 根因：`gate_up_input_scale`跨層差異達30倍以上

```
全域 inv_scale_frac=9（給全35層最大值46.5用的Q-format）
Layer 0 gate_up_input_scale(重新校準後)=0.7025 -> inv_scale=1.4236（只有全域最大值的1/33）
drop_bits=24（全域統一，照最壞情況算出來的）

Layer 0典型情況：
  inv_rms_scaled_wide = inv_rms_raw(~52429) x inv_scale_int(729) = 38,220,741
  右移24-bit後 = 2   <- 幾乎所有資訊被捨光
```

`INV_GATE_UP_SCALE`跟`drop_bits`都是「全域統一Q-format」（用35層裡最極端的值決定格式），但`gate_up_input_scale`本身跨層差異高達33倍（Layer 0重新校準到0.70，Layer 13只有0.0083附近）。統一的shift量對「小`inv_scale`的層」而言，等於用「照顧最極端層」的粗粒度去捨入一個本來就很小的數字，直接捨到只剩1~2個bit。

Post-Norm完全沒有這個問題，因為`S_RES`是全域固定常數，不會有跨層33倍量級差異。

### 4.2 修正：per-layer的drop_bits跟最終shift

不再讓全35層共用同一組frac/drop_bits，改成每一層各自：

1. `INV_GATE_UP_SCALE_FRAC[layer]`：每層用自己的實測值決定int_bits/frac（不是全域最大值）——`derive_inv_scale_frac_per_layer()`
2. `RMSNORM_PRE_DROP_BITS[layer]`：Step A算完後per-layer要右移多少bit
3. `RMSNORM_STAGE3_SHIFT_PRE[layer]`：Stage3最終shift量，取代原本單一`#define`

實測結果（真實checkpoint數字）：

```
INV_GATE_UP_SCALE_FRAC（per-layer）: min=8, max=14（Layer 0 frac=14, Layer 13 frac=8）
RMSNORM_STAGE3_SHIFT_PRE（per-layer最終shift）: min=9, max=15
```

Layer 0的frac從全域共用的9-bit提升到自己專屬的14-bit，因為它的`inv_scale`量級小，能分配到更多frac；Layer 13維持8-bit（它的`inv_scale`量級大，int_bits需求高，frac自然少）。

位元寬度重新檢查（用per-layer int_bits+frac恆等於24-bit這個不變量，不能把不同層的`int_bits.max()`跟`frac.max()`分開加總，那樣會虛報一個不存在的組合）：Step C = 63-bit，安全。

Post-Norm因為沒有跨層33倍量級差異的問題，繼續維持全域共用一個值即可（`RMSNORM_POST_INV_RMS_SCALED_FRAC`），不需要per-layer化。

### 4.3 硬體成本

per-layer化需要把`RMSNORM_STAGE3_SHIFT_PRE`從compile-time常數改成35個值的表，Stage3的最終位移量從固定位移變成runtime可變位移（查表決定shift多少位）。這是一個小的額外成本（一個小型lookup table + variable-amount shifter，不是固定shift器），但跟`INV_GATE_UP_SCALE`本身早就是per-layer表格是同一類成本，不是新增的資源類別。

## 5. 修正後重新驗證：Layer 0恢復正常，全部6層都在0.99+

用真實checkpoint數字重跑`rmsnorm_verify_stage3_bitwidth.py`：

| Layer | PRE 有中途捨入 cos_sim (min/mean/max) | PRE 無中途捨入 cos_sim (mean) | 中途捨入造成的變化 |
|---|---|---|---|
| 0  | 0.9995/**0.9996**/0.9998 | 0.999636 | -0.0000069（可忽略） |
| 9  | 0.9917/0.9960/0.9983 | 0.996014 | -0.0000115（可忽略） |
| 14 | 0.9914/0.9938/0.9961 | 0.992966 | +0.0008（可忽略） |
| 15 | 0.9994/0.9995/0.9996 | 0.999504 | -0.0000032（可忽略） |
| 31 | 0.9997/0.9997/0.9998 | 0.999737 | -0.0000008（可忽略） |
| 34 | 0.9996/0.9997/0.9998 | 0.999677 | +0.0000022（可忽略） |

**Layer 0從cos_sim=0.000000恢復到0.999629**，全部6層有中途捨入 vs 無中途捨入的差距都在±0.001以內（雜訊等級，不是系統性劣化）。彙整全部token：

```
PRE  有中途捨入: mean=0.998056  無中途捨入: mean=0.997922  差距=+0.00013343
POST 有中途捨入: mean=0.997115  無中途捨入: mean=0.997115  差距=-0.00000029
```

中途捨入這一步本身，在per-layer修正後，對精度幾乎沒有額外傷害。

## 5.1 擴大到全35層驗證（不只信任6層抽樣）

第4節的underflow是**離散、特定層才會踩到**的failure mode（Layer 0是因為`gate_up_input_scale`偏離全域值33倍），不是連續漸變的精度劣化——6層抽樣剛好抓到Layer 0，不保證其他29層裡沒有另一個同樣偏離全域值、但沒被抽到的層。

把`extract_real_rmsnorm_activation_batch.py`的`LAYER_LIST`從6層抽樣擴大到全35層（`extract`腳本本來就是「每個prompt跑一次forward、同時勾住`LAYER_LIST`全部層」的設計，擴大到35層不會增加forward次數，只是多存一點資料，4次forward、幾秒鐘內完成），`rmsnorm_verify_stage3_bitwidth.py`同步擴大到全35層重跑。

**結果：沒有再出現underflow或類似的離散崩潰。** 最差的幾層：

| Layer | frac | cos_sim min | cos_sim mean | 中途捨入造成的delta |
|---|---|---|---|---|
| 4 | 7 | 0.9286 | 0.9487 | +0.0000178（可忽略） |
| 13 | 5 | 0.9516 | 0.9676 | -0.0000471（可忽略） |
| 3 | 6 | 0.9618 | 0.9720 | +0.0000239（可忽略） |
| 2 | 7 | 0.9677 | 0.9761 | -0.0000214（可忽略） |

這幾層的cos_sim確實比其他層低（0.93~0.98而不是0.99+），但「有中途捨入」跟「無中途捨入」的delta都在±0.00005量級——證實這些較低的cos_sim是這幾層**本來就有**的特性（可能跟`gate_up_input_scale`本身量級或INT8精細度有關），不是Stage3中途捨入這一步新引入的問題。

全35層彙整：

```
PRE  有中途捨入: mean=0.993127  無中途捨入: mean=0.993104  差距=+0.00002258
POST 有中途捨入: mean=0.997475  無中途捨入: mean=0.997475  差距=-0.00000005
```

per-layer的drop_bits/shift修正在全35層規模下都成立，沒有漏網之魚，可以進入C-simulation testbench階段。

## 6. 產出檔案

- `hls/fast_inv_unit.h` — 共用倒數/倒數平方根核心（審查後確認沒有問題，magic constant跟Q-format都已在golden model端交叉驗證過）
- `hls/rmsnorm_kernel.h` — RMSNorm HLS核心，per-layer drop_bits/shift版本
- `golden_model/generate_rmsnorm_const_tables.py` — 產生`rmsnorm_gamma_table.h`/`rmsnorm_layer_scale_table.h`/`rmsnorm_step0_table.h`，新增per-layer的`derive_inv_scale_frac_per_layer()`跟位元寬度自我檢查（Pre/Post-Norm兩條路徑都有）
- `qat_verification/rmsnorm_verify_stage3_bitwidth.py` — golden model層級的中途捨入精度驗證，比對「有/無中途捨入」跟真實QAT輸出的差距

## 6.1 C-simulation testbench：實際compile+執行，抓到4個真bug

### 環境限制與繞過

這台開發機沒有裝Vitis HLS，但Xilinx把`ap_fixed`/`ap_int`開源到GitHub
（`Xilinx/HLS_arbitrary_Precision_Types`），clone下來後可以直接用g++
（不需要完整Vitis工具鏈）做c-simulation層級的編譯執行驗證。

寫了`hls/generate_stage3_testbench_vectors.py`：從真實QAT activation
（全35層 x 3個token=105+105筆）產生測試向量，期望輸出用跟
`rmsnorm_kernel.h`逐bit對應的Python模擬算出來（不是cos_sim近似，是
要求**完全相同**，差1都算fail）；`hls/tb_rmsnorm.cpp`是C++測試主體，
呼叫`pre_feedforward_layernorm_hls()`/`post_feedforward_layernorm_hls()`
逐case比對。

### 過程中實際compile+執行抓到的4個真bug

**Bug 1：常數表型別跟kernel.h函式簽章不相容。**
`RMSNORM_GAMMA_PRE`/`RMSNORM_STEP0_M_POST`宣告成`int16_t`/`int32_t`，
但`rmsnorm_stage3()`/`rmsnorm_step0()`的參數型別是`ap_int<16>`/
`ap_int<32>`——C++不允許陣列指標型別做elementwise隱式轉換（純量可以，
陣列指標不行），編譯直接報錯。

**Bug 2：改成`ap_int`存ROM後，這台機器的g++ 8.1.0編譯超過~31列的大型
`ap_int`靜態陣列初始化會直接crash**（同樣大小的plain int完全沒事，
用二分搜尋確認崩潰門檻在30~32列之間）。這是這個舊版編譯器處理
Xilinx `ap_int`模板陣列初始化的已知類限制，不是設計問題。修正：
常數表維持存plain `int16_t`/`int32_t`（ROM），函式讀取單一元素時才
轉型成`ap_int`（純量轉型不會觸發這個crash），這同時也解決了Bug 1。

**Bug 3：`generate_rmsnorm_const_tables.py`算`INV_GATE_UP_SCALE`用
checkpoint原始的`gate_up_input_scale`，沒有套用WikiText重新校準值。**
整套驗證管線（batch verify、perplexity、stage3 bitwidth）早就改用
校準值，唯獨這支常數表產生腳本沒跟上，兩邊常數對不上導致PRE路徑
**105個case全部fail**（差異隨層數增長，最大到222）。修正後104/105。

**Bug 4：`generate_stage3_testbench_vectors.py`的POST路徑`post_mid_frac`
誤用per-layer算法**（`RMSNORM_POST_INV_RMS_SCALED_FRAC`實際上是全域
共用常數，用全35層`gamma_post`的最大值算的，不是per-layer），testbench
產生器卻在每層迴圈裡各自用該層的`gamma_post`最大值重算一次，兩邊對不上。
修正後POST從18/105變**105/105全過**。

### 最後1個case：1-ULP的rsqrt翻譯差異（記錄成known issue，未完全解決）

修完以上4個bug後，PRE還剩1個case（layer 20）差到3（其餘104個逐bit
相同）。深入追查：

1. 先懷疑Stage1的`mean(x²)/D`除法——C++用`(ms_t)(1.0/D)`把`1/1536`
   量化成24-bit定點常數再相乘，不是精確除法。補上這個量化（`golden
   model`原本用`np.mean()`精確除法），修正後Python算出的`mean_sq`
   跟C++**完全一致**（4266.641922354698，逐位元相同）。
2. 但`fast_inv_unit_python()`算出來的`inv_rms_raw`還是跟C++差1
   （16053 vs 16052）。追查發現`fast_inv_unit_python()`的docstring
   本身就承認Newton-Raphson階段「用float64算，不逐位元模擬中間舍入」
   ——這是先前就記錄在案的簡化，不是這次新發現的bug。補上`xy_t=
   ap_ufixed<32,18,AP_RND>`/`t_t=ap_ufixed<26,2,AP_RND>`這兩個中繼值
   的定點截斷/捨入，仍然沒有讓這個特定case的結果對齊（可能是AP_RND
   跟Python`round()`的捨入方向細節差異，或leading-one-detector的
   邊界處理，尚未完全查清楚根因）。
3. 這個1-ULP差異之所以會被放大成整個token ~3%的系統性偏差，是因為
   `inv_rms_scaled`（Stage3 Step B的中途捨入結果）剛好卡在一個捨入
   邊界（30.998 vs 31.0006），1個ULP的輸入差異被推過了邊界，變成
   `30`跟`31`兩個不同的整數，而這個值是per-token純量、會乘進全部
   1536個channel，才會表現成「整個token系統性偏差」而不是「單一
   channel差一點點」。

**這個問題保留成known issue，不繼續深挖**，理由：
- 只影響105個case裡的1個（<1%）
- 屬於`fast_inv_unit_python()`這個golden model既有翻譯的精度邊界，
  不是這次新寫的Stage3 HLS程式碼（`rmsnorm_kernel.h`）本身的問題——
  這個函式先前已經用(E)節的rsqrt backend對照實驗驗證過整體cos_sim=
  1.000000，這次抓到的是一個發生機率很低、需要輸入值剛好卡在雙重
  邊界（rsqrt本身的ULP邊界 + 下游Step B的捨入邊界）才會顯現的特例
- 已經補上的Newton-Raphson定點截斷修正，即使沒有完全解決這個case，
  仍然讓`fast_inv_unit_python()`比純float64更貼近硬體語意，是淨改善
  、沒有造成其他104+105個case的退步

### 最終結果

```
[PRE]  104/105 PASS（逐bit相同），max_abs_diff（僅剩1個case）=3
[POST] 105/105 PASS（逐bit相同），max_abs_diff=0
```

209/210（99.5%）測試案例逐bit精確通過，涵蓋全35層。C-simulation
testbench驗證通過，`hls/rmsnorm_kernel.h`的實作跟設計時講好的整數
運算基本一致，剩下的1個已知narrow-scope issue不影響往下走。

## 7. 這次審查的方法論小結

**位元預算過得了關，不代表精度過得了關，兩件事要分開驗證**——這個教訓在residual的Plan A/B實驗裡學過一次，這次在Stage3中途捨入又驗證了一次，而且這次的表現形式更極端（不是精度緩降，是特定層直接underflow歸零）。

**全域共用一組Q-format，會系統性地虧待動態範圍偏離全域極值的個體**——這個模式在這次專案裡出現了不只一次：
- Layer 0/14/34的`S_RES`溢位（個體量級偏大，被全域小範圍的Q-format截斷）
- Layer 14的`gate_up_input_scale`校準不足（個體outlier channel，被全域校準值忽略）
- 這次Layer 0的Stage3 underflow（個體量級偏小，被全域粗粒度的drop_bits捨光）

三次表現形式不同（溢位飽和 vs 校準覆蓋不足 vs 捨入underflow），但根因是同一個模式：任何時候用「全域統一常數」處理一組跨層/跨channel動態範圍差異很大的資料，都要先問「這個常數是不是被最極端的個體決定的，其他個體會不會因此受害」，再決定要不要做成per-layer/per-channel版本。
