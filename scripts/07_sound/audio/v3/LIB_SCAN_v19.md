# LIB_SCAN_v19: 本地音效库扫描报告 (浙江农林大学 108 s 宣传片的声音设计)

扫描目标: 让鸟叫和环境声更好地融入画面; 排除乌鸦叫; 找到更优美的音效。对照对象是 v18 声音设计的用法。

---

## 0. 结论速览

1. **"乌鸦"的来源找到了, 有两处:**
   - **Country Day** 里有 4 串真正的乌鸦连叫 (频谱上是 1-4 kHz 的粗糙谐波堆叠, 每声约 0.2 s, 间隔约 0.6 s): 文件 **11.4-14.1 / 21.2-25.1 / 32.1-35.5 / 42.1-45.8 s**; 另有远处狗叫 **38.4-39.7, 46.2-51.0 s**, 和低沉的 "hoo" 声 **52.5-56.9 s**。
   - **Bird Seagull** 整个文件都是海鸥/鸭样叫声, 开头 0.15-3.0 s 是一串 "哈-哈-哈" 笑鸥, 听感就像乌鸦。v18 在 33.8 s、70.2 s 用的就是这一段 (0-1.2 s / 0-1.0 s), 双耳版 48.0 s 的环绕鸟也是它 (0-6 s)。
   - 按 v18 的 film_time 对位, 乌鸦实际出现在影片 **32.1-35.0 s** (图书馆落成长和弦下面), **72.7-75.4 s** (校训石), **82.5-86.4 s** (山谷中的校园); 狗叫在 **46.9-51.0 s**; 5.1 后环绕声道还会在 51.9-53.5、62.8-66.2、72.8-76.5 s 再出乌鸦。详见 1.1 节。
2. **Forest Day 没有乌鸦/海鸥/公鸡/狗/人声/交通** (全部窗口 crow ≤ 0.06), 底噪极低, 但鸟群很密, 有几段尖锐的下行颤音被 CLAP 判为 "jungle/parrot"。用它的干净区段即可 (1.3 节)。
3. **最优美的鸟鸣素材:** Birds FX 02 的 8 段歌唱 (CLAP good 0.54-0.71, harsh ≤ 0.17), Birds FX 01 的口哨样鸣叫, Bird Sparrow 的单句。**注意: Birds FX 01/02 有很强的 50 Hz 交流哼声和 150-235 Hz 低频音调, 必须高通 ≥ 1.2 kHz** (鸟声都在 2 kHz 以上, 高通不伤鸟)。
4. **Wind 4 (现用) 被限制在约 3.5 kHz 以下** (4-10 kHz 带只有 -70 dB), 闷, CLAP 判为 "microphone wind buffeting"。更通透的风: Wind 5、Wind 6、Wind 8 (竹叶风, 安静)。不过 Wind 4 的阵风正好是最柔的天然 whoosh。
5. **Forest Stream River Water 01 本身干净, 但 57.0-62.5 s 是渐出, 62.5-64.85 s 是数字静音。** v18 "一水" (影片 57.43-59.6 s) 恰好读到文件 55.6-60.6 s 的渐出段, 水声在镜头内掉约 10 dB。
6. **现用 whoosh (Whoosh 17 / Whoosh 02) 被 CLAP 判为 "a harsh aggressive whoosh" 0.84 / 0.91**, 建议换成柔和的 (3.6 节)。
7. 过渡/点缀首选: Bell Small, Bell Clang Resonance Low.M, Harp Transition 1 (升 2 半音), Harp Transition 2 (升 1 半音), Chimes 1, Shimmer Motion, Bell Church 2 (D4, 远寺钟感), MAGMisc_Magic Christmas Bells 2 (玻璃碎光), Wing Flaps Multiple Birds Fly Away, Eagle Wing Flaps.M, A4 Printing Paper Rattle Page Turn Tail (保留)。

---

## 方法

- CLAP 耳朵: `scripts/07_sound/naiwa_engine/audio/v8/clap_tool.py` (laion/larger_clap_general), 文件经 ffmpeg 解码为 48 kHz float, 5.1 文件用 eng.fold6 同一公式折成单声道。
- **每个窗口先归一到 -24 dBFS RMS 再嵌入** (Forest 01 之类 -60 dB 的安静文件不归一时 CLAP 判断会漂; 实际混音里床都会被推到目标电平, 归一更接近听感)。
- 床/候选: 2.0 s 窗, 1.0 s 步长; 关键文件再做 1.0 s 窗 / 0.5 s 步长细扫。一次性音效: 2.0 s 窗 0.5 s 步长 + 自动分段 (每个 take 单独嵌入)。
- 概率 = softmax(20 x cos) (与 clap_tool.score 相同), 环境标签集 29 个:
  - GOOD: melodious songbirds singing / gentle birdsong in a quiet forest / a blackbird or thrush singing a melodious song / small birds chirping softly
  - BAD: a crow cawing / seagulls crying / a rooster or chicken / ducks quacking or geese honking / a parrot or jungle bird screeching / a loud harsh bird squawk / an owl hooting / traffic or engine / an airplane flying overhead / people talking / children shouting / dog barking / electrical hum / cows or farm animals / monkeys calling in a jungle
  - 中性: insects buzzing / crickets chirping / wind / wind rustling leaves / running water stream / calm lake water lapping / frogs croaking / silence, faint room tone / leaves rustling close up / a gentle breeze in the trees
- **校准** (已知文件): Bird Crow 有乌鸦的窗 crow = 0.42-0.48; Bird Rooster 0.65-0.89; Bird Duck 0.47-0.62; Bird Geese 0.62-0.83; Dog Bark Small 0.82-0.91; Bird Owl 0.84。判定阈值: 2 s 窗 crow ≥ 0.12、1 s 窗 ≥ 0.15 记为可疑, 然后 **用频谱图人工复核** (乌鸦 = 1-4 kHz 宽带粗糙谐波, 0.2 s 一声, 约 0.6 s 间隔连叫)。
- "a parrot or jungle bird screeching" 在所有密集鸟鸣里都有 0.15-0.25 的底数, 只作相对比较。HARSH = parrot + harsh squawk + monkeys 三项之和, ≥ 0.40 记为偏尖。
- 音高: FFT 峰值 (onset 后 30 ms 起 0.8 s), 另做 chroma (音级能量) 并计算与 D 大调五声 (D E F# A B) 的吻合度; v18 的 glow 音是 D6 / A5 / F#5, 所以给出"移到 D 调需要的 speed()"。speed 比 = 2^(半音/12), r > 1 变快变高。
- 电平: pk = 窗口内所有声道的采样峰值 dBFS; rms = 单声道 RMS dBFS; floor10 = 100 ms 块 RMS 的第 10 百分位。

---

## 1. 现用 5 个素材逐窗扫描

### 1.1 Country Day (61.33 s, 立体声): 有乌鸦和狗, 建议整体弃用

| 文件时间 (s) | 内容 | CLAP 证据 (1 s 窗 / 2 s 窗) | 结论 |
|---|---|---|---|
| 0.0-0.5 | 文件渐入 (-68 dB) | | 接缝处会掉坑 |
| 0.5-11.3 | 蟋蟀 + 零星小鸟 | crickets 0.27-0.42, crow ≤ 0.09 | 干净, 但嘶声偏多 (>10 kHz 带 -43.9 dB) |
| **11.4-14.1** | **乌鸦连叫 5 声** (11.5/12.1/12.7/13.3/13.9) | crow 0.18-0.32 | **避开** |
| 14.2-20.4 | 一段鸣禽颤音 (13.5-15.8), 之后蟋蟀 | good 0.35, crow < 0.1 | 干净 |
| **20.5-25.1** | **乌鸦连叫** (21.3-23.5, 24.1-25.0 夹鸣禽) | crow 0.17-0.32 | **避开** |
| 25.2-31.4 | 蟋蟀为主, 30-31 有小鸟 | crickets 0.3-0.44 | 干净 |
| **31.5-35.5** | **乌鸦连叫** (32.2-35.4, 间隔 0.6 s) | crow 0.15-**0.41** (文件最高, 32.5 s) | **避开** |
| 35.5-36.5 | 低沉叫声 | owl 0.21 | 避开 |
| 37.0-40.0 | 低频短叫 (38.5, 39.0, 39.5) | dog 0.18-0.33, cows 0.20-0.29 | 避开 |
| 39.3-42.0 | 鸣禽滑音 (偏尖) | harsh 0.40 | 勉强 |
| **42.0-46.0** | **乌鸦连叫** | crow 0.17-0.38 | **避开** |
| **45.5-51.0** | **远处狗叫** | dog 0.27-0.67 (2 s 窗 0.79 @ 47) | **避开** |
| 52.5-56.9 | 每秒一声低沉 "hoo" (<1 kHz) | owl 0.22-0.53 | 避开 |
| 56.9-59.6 | 一段漂亮的鸣禽颤音 (57.2-59.5) | crow < 0.1 | 干净 |
| 59.5-61.33 | 渐出 (-41 到 -49 dB) | | 接缝掉坑 |

整个文件 good 均值只有 0.14, 不是"优美鸟鸣"素材。

**v18 中实际落点** (bed() 用 film_time: 文件位置 = 影片时间 mod 61.33; 可听区间含 lead 和 pre/post):

| v18 调用 | 可听影片区间 | 读到的文件区间 | 问题 |
|---|---|---|---|
| jbed 27.4-34.2 (lead 0.8) "East Lake site" | 26.35-35.2 | 26.35-35.2 | **乌鸦 32.1-35.2** (压在 32.75 木鱼 + 32.8 图书馆长和弦上, 后 1.5 s 在淡出) |
| jbed 48.3-52.54 "campus from above" | 46.85-53.5 | 46.85-53.5 | **狗叫 46.9-51.0**, hoo 52.5-53.5 |
| jbed 59.6-66.8 "misty library at dawn" | 58.15-67.8 | 58.15-61.33 + 0-6.5 | 无乌鸦; 但 **60.5-62.3 s 文件尾/头接缝掉坑约 20 dB** (-49 到 -68 dB) |
| jbed 73.0-80.0 "motto stone" | 71.55-81.0 | 10.2-19.7 | **乌鸦 72.7-75.4** |
| jbed 80.0-87.8 "campus in its valley" | 78.55-88.8 | 17.2-27.5 | **乌鸦 82.5-86.4** |
| 5.1 后环绕 (rear = 文件位置 + 30.67 s, -4 dB, LPF 6 kHz, 仅 5.1) | | | 乌鸦另在 51.9-53.5, 62.8-66.2, 72.8-76.5; 狗叫 76.9-81.7; hoo 83.2-87.6 |

### 1.2 Bird Seagull (31.87 s): 全文件不可用

- 每一个窗口都被污染: gull 0.23-0.41, duck 0.36-0.54, rooster 0.12-0.37, harsh 0.40-0.58, crow 最高 0.23 (7.5 s, 1 s 窗); good 均值 0.02。
- 频谱: 0.15-3.0 s 是一串 "哈-哈-哈" 笑鸥 (1-4 kHz 谐波堆叠, 约 0.25 s 间隔), 3-8 s 起是群鸥。
- **没有任何干净窗口。** v18 用法全部要删: one 33.8 s (0-1.2 s, 正是笑鸥), one 70.2 s (0-1.0 s), 双耳 orb 48.0 s (0-6 s)。

### 1.3 Forest Day (64.7 s): 无乌鸦, 可用其干净区段

- 污染标签最大值 (全文件): crow 0.06, gull 0.06, rooster 0.15 (27.5 s, 1 s 窗), owl 0.15, dog 0.00, traffic 0.01, people 0.01, hum 0.00。good 均值 0.41 (所有床里最高)。
- 频谱上两种主角:
  - (a) 4 kHz 左右的长滑音口哨, 很美: 1.8-3.0, 16.0-17.0, 21.0-22.0, 24.6-26.0, 32.3-33.5, 39.0-40.0, 46.6-48.0, 54.5-55.8, 61.6-63.0 s;
  - (b) 5-11 kHz 阶梯式下行颤音, 偏尖: 4.8-6.5, 16.0-17.5, 27.3-28.9, 40.0-41.0, 43.0-44.3, 51.4-53.0, 60.3-61.6 s (大约每 11 s 重复一次, 循环使用时容易被听出)。
  - 底下是 2-3 kHz 连续群鸣。
- 偏尖 (HARSH ≥ 0.40) 的 2 s 窗: 4-8, 9-11, 16-18, 27-29, 36-38, 42-45, 51-53, 58-61 s。
- **干净区段 (1 s 粒度):** 0.5-4, 11-16, 18-27, 29-36, 38-42, 45-51, 53-58, 61-64 s。
- 最佳 3 s 窗 (直接嵌入复核): **53-56** (good 0.52, harsh 0.28, pk -9.1, rms -27.6), **11-14** (good 0.58, harsh 0.27, pk -3.9, rms -22.1), **30-33** (good 0.50, harsh 0.30)。长段 18-27 较安静 (rms -32.1)。
- 单句: 54.4-55.9 (good 0.59, harsh 0.23), 46.5-48.1 (0.47 / 0.32), 20.9-22.1 (0.40 / 0.22)。
- v18 中: 10.0-13.0 床读 9.45-13.75 (含 9-11 偏尖), 13.0-16.6 读 11.55-17.35 (含 16-17), 34.2-37.8 读 32.75-38.55 (含 36-38), 37.8-42.6 读 36.35-43.35, 52.54-56.07 读 51.1-56.8 (含 51-53), **69.4-73.0 读 3.25-9.3 (正好是 4.8-6.5 的尖颤音)**, 92.0-93.8 读 26.45-30.1 (含 27-29)。都不是乌鸦, 只是不够"优美"。bed() 的 `a=` 参数会改变对位 (文件位置 = a + 影片时间 mod (时长 - a)), 可借此对到干净区段。one("Forest Day", 17.1, 0-0.5) 读的是文件渐入 (0-0.5 s 只有 -53 dB)。

### 1.4 Wind 4 (59.9 s): 干净但闷

- 无鸟、无人声。唯一可疑: 36-38 s 低频隆隆 (traffic/airplane 0.19, 1 s 窗 0.29); hum ≤ 0.10。
- **频带被限制在约 3.5 kHz 以下**: 120 Hz-1 kHz -23.4 dB, 1-4 kHz -38.1, 4-10 kHz -69.7, >10 kHz -76.0。CLAP 风标签集: microphone wind buffeting 0.42 (均值), wind in bamboo 0.34。听感闷、像贴麦风噪, 不适合"竹林风"(v18 42.6-48.3 s)。
- 头 0-1.0 s 渐入, 尾 57.5-59.9 s 渐出。

### 1.5 Forest Stream River Water 01 (64.85 s, 5.1): 干净, 但尾巴是渐出

- 每个窗口 running water 0.84-0.97, 无鸟无人声 (尾部数字静音里的 crow 0.17-0.19 是静音伪迹)。
- CLAP 水标签集: "a fast rushing river" 0.51-0.61, "a small babbling brook" 0.18-0.27, 偏急流白噪。rms -38.0, 极平稳 (0.5 s 块标准差 0.4 dB), LF<120 Hz -84 dB, >10 kHz -63 dB。
- **头 0-1.0 s 渐入 (-78 / -46 dB); 57.0-62.5 s 渐出 (-38 到 -67 dB); 62.5-64.85 s 数字静音。安全区 1.0-56.5 s。**
- v18: one 29.0 s (a=0-1.6) 读的是渐入段; **jbed 57.43-59.6 "一水" 读文件 55.6-60.6, 正在渐出, 镜头内掉约 10 dB**; jbed 87.8-90.8 读 21.8-26.95, 没问题。

---

## 2. 替换/补充床与鸟鸣

### 2.1 鸟鸣 (按优美度排序)

**A. Birds FX 02 (54.31 s, 立体声): 最美, 但必须高通**

歌唱段 (每段单独嵌入复核, 原始 / HPF 1.2 kHz 两种都测过):

| 窗口 (s) | good | harsh | 其它 | pk / rms (dBFS) |
|---|---|---|---|---|
| **0.00-2.80** | **0.71** | 0.13 | blackbird/thrush 0.58 | -2.4 / -16.7 |
| **47.90-49.90** | 0.65 | 0.12 | | -1.9 / -17.8 |
| **14.70-17.20** | 0.61 | 0.09 | | -2.1 / -17.2 |
| **23.50-26.00** | 0.61 | 0.13 | | -3.4 / -17.7 |
| **30.90-33.30** | 0.60 | 0.16 | | -0.8 / -16.9 |
| 39.90-42.10 | 0.54 | 0.13 | | -1.9 / -16.5 |
| 7.90-9.50 | 0.41 | 0.07 | 较弱 | -2.5 / -17.9 |
| 52.70-54.30 | 0.37 (HPF 后 0.51) | 0.15 | | -1.2 / -16.5 |

3 s 窗 (自动挑选 + 复核): 0-3 (good 0.75), 14-17 (0.72), 47-50 (0.66)。

**避开:** 5-8, 28.5-30, 35.0-40.5, 42-47.5, 49.5-53 s。这些是单音 "chip" 连叫 (约 0.6 s 一声, 2.5-5 kHz), CLAP 判 rooster/chicken 0.20-0.54, cows 0.21, 35-39.5 整段 harsh 0.42, 像母鸡咯咯。

底噪 (重要): 50 Hz 哼声 (高出邻频 38 dB) + 148/190/205/220/235 Hz 低频音调 (突出 50-80 dB); LF<120 Hz -35 dB, 120 Hz-1 kHz -25 dB。**HPF ≥ 1.2 kHz 必须**; 高通后 1-4 kHz 底 -38.7, 4-10 kHz 底 -48 到 -50 dB, 歌声 p90 约 -13 到 -15 dB, 信噪比约 30 dB。

**B. Birds FX 01 (7.59 s): 纯净口哨样鸣叫, 点缀用**

- 0.00-1.45: 四声下行口哨 (4-5 kHz), good 0.58, harsh 0.10, pk -0.9 (被低频顶的), rms -14.0
- 3.55-4.10: 上滑哨, good 0.51
- 4.25-6.95: "tee-tyu" 双音四次, good 0.44, harsh 0.12
- 全段 0-7.5: good 0.66, harsh 0.16
- 底噪: **49.9 Hz 哼声 -24.7 dBFS (突出 62 dB)** + 谐波, LF<120 Hz -22.8 dB, 120 Hz-1 kHz -20.8 dB, 比鸟还响。**HPF 1.5-2 kHz 必须**, 之后 4-10 kHz 底约 -42 dB。

**C. Bird Sparrow (12.95 s): 干净单句**

- 同一乐句重复 3 次, 完全相同: **0.00-2.05**, 5.00-7.05, 9.05-11.05。
- good 0.43-0.52, harsh 0.39-0.43 (偏亮, 4-7 kHz), pk -6.9, rms -19.3。
- 句间是数字静音 (LF -65 dB, 120 Hz-1 kHz -84 dB), 无任何底噪。只适合单次点缀, 不要重复同一句。

**D. Forest Day: 见 1.3 节的干净区段。** 底噪最低的鸟床 (floor10 -42 dB, LF -61 dB, >10 kHz -63 dB)。

**E. Country Day 干净区段 (如果非要保留):** 0.5-11.3, 14.2-20.4, 25.2-31.4, 56.9-59.6 s。good 只有 0.19-0.28, 不美。

**不推荐的鸟/环境 (原因):**
- Bird Blackbird: 其实是 "chink" 警报叫 + 每 0.75 s 一次宽带咔哒, CLAP frogs 0.47-0.81, harsh 最高 0.46
- Bird Dove: 低频咕咕 + 6.0/6.5/7.1 s 有咔哒, CLAP owl/rooster
- Bird Hummingbird: 虫翅嗡嗡 + rooster 样叫声
- Forest 01: 极安静 (-55 到 -62 dB), 一只鸟每 0.5 s 一声 3 kHz chip, 多窗 crow 0.12-0.22, 41.5 / 55.8 s 有粗哑大叫
- Jungle Morning 1 (harsh/duck/gull 遍布), Jungle Morning 2 (5-6 kHz 虫鸣脉冲墙), Jungle Day (harsh)
- AMBSubn_Ambience, Forest Crickets, Birds, Connecticut 02: 6.5 kHz 一条恒定纺织娘/蟋蟀声带, 鸟稀疏, 归一后 harsh 0.44; 只有 2-8 s 可当夏夜虫鸣床
- Marsh Day (蛙 + 嗡鸣叫), Marsh 01 (虫), Farm 01 (首尾公鸡), Country Night / Forest Night (夜虫), Beach FX 01 (海浪), TOONMisc_Bird Flutes 3 (卡通鸟笛), AMBTrop (热带夜), AMBPark_Berlin (交通底噪反复出现)

### 2.2 水

| 文件 / 窗口 (s) | CLAP (水标签集) | rms / floor10 | 备注 |
|---|---|---|---|
| **Water Stream River 01 7-10** | **babbling brook 0.50**, river 0.25 | -43.9 / -44.5 | 最像小溪, 极平稳; LF -75, >10k -65 |
| Water Stream River 01 17-25 / 35-43 | river 0.44-0.48, brook 0.26-0.28 | -43.7 / -44.4 | 安全区 1.0-59.5 (59.8 起渐出, 62.3 起数字静音) |
| Forest Stream River Water 01 17-25 / 30-38 / 42-50 | river 0.51-0.61 | -38.0 / -38.8 | 现用; 安全区 1.0-56.5 |
| Water River 2 15-23 / 28-36 / 51-59 | brook 0.28-0.34, fountain 0.29-0.38 | -30.2 / -32.2 | 更近更亮; 120 Hz 小哼 (带电平 -76 dB, 听不到); 60.5 起渐出 |
| Water Lake 1 19-23 | **lapping 0.59** | -28.7 / -37.0 | 22.5, 24.0, 26.5, 28.3 s 有拍岸浪 |
| Water Lake 1 46-54 | ocean waves 0.50, lapping 0.43 | -32.4 / -42.4 | 偏海浪 |
| WATRLap_Summer Tennessee Lake Dock ... 10-18 | lapping 0.30 (混合) | -33.5 / -36.9 | 湖边码头, 最"小湖"; LF -53 dB, 建议 HPF 80-100 Hz |
| WATRLap ... 37-40 | brook 0.37, fountain 0.35 | -39.5 / -41.6 | 安静 |
| Water Lake 2 / 3 | ocean waves 0.43-0.59 > lapping | | 对东湖来说太大 |
| Water River 1 | fast river 0.49-0.53, -19 dB | | 只适合大水 |

WATRLap 干净区: 0-34, 37-43, 45-47, 49-62, 64-102 s。33.0-33.8 有一次宽带拍水; 35-47 s 有远处 3.5-5 kHz 小鸟 chip (CLAP rooster 0.22-0.31, 在 35-37 和 43-45 s)。

### 2.3 风

| 文件 | 最佳窗口 (s) | CLAP (风标签集) | 频带特征 | 噪声/问题 |
|---|---|---|---|---|
| **Wind 5** | **23-31, 33-41, 52-60** | wind in bamboo 0.92-0.94 | 宽带, 4-10k -37, >10k -47, 平稳 (块标准差 1.0-1.2 dB) | LF<120 -30 dB, HPF 120 Hz; 31 s 一次 harsh 0.34 |
| **Wind 8** | **1-7, 10-14, 35-43**; 竹叶风 **5-13** (bamboo leaves 0.80) | wind in bamboo 0.86-0.89 | 安静 (-30 到 -35 dB), 有叶片沙沙 | **60 Hz 哼 -50 dBFS (突出 66 dB)**, HPF 100 Hz; 0-0.5 渐入 |
| Wind 6 | 29-35, 17-21, 37-41 | wind in bamboo 0.77-0.84 | 通透嘶声 (1-4k -22), LF -45 | 其它段 buffeting 最高 0.57, 只用列出的窗 |
| Wind 3 | 10-13, 16-19, 29-32 | wind in bamboo 0.78-0.85 | 偏暗 (4-10k -47) | 26-28 s 像飞机 0.24; 只有 32 s 长 |
| Wind 7 | 4-9, 19-25 | wind in bamboo 0.72-0.73, leaves 0.11-0.18 | 叶 + 水样沙沙, 安静 | **60 Hz 哼 -47.7 dBFS (突出 74 dB) + 120 Hz**, HPF 150 Hz |
| Wind 1 | 39-42 | wind in bamboo 0.82 | 阵风多, 夹 "running water" | buffeting 均值 0.14 |
| Wind 4 (现用) | 全段干净 | buffeting 0.42 | <3.5 kHz, 闷 | 36-38 低频隆隆 |
| Wind 2 | 不推荐 | howling 0.23, rumble 0.17 | 1-4k -50 (极暗) | 50/60 Hz 谐波突出 20-25 dB |
| Wind 9 | 不推荐 | buffeting 0.60 | | 贴麦风噪 |

### 2.4 树叶

近距拟音, 适合做点缀, 不适合做床 (用床的标签集时 CLAP 判为 "branches cracking" 0.63-0.66, 偏脆):
- Leaves Bush Rustling.M (41.4 s, 单声道): **6.0-8.0** (leaves rustling 0.80, pk -10.2, rms -33.3), **14.5-16.5** (0.79, pk -10.7, rms -29.1), 0.49-10.49 (0.82)
- Rustling Branches and Leaves.M: 0.5-2.5 (0.79, pk -22.2, rms -46.8), 12.36-14.36 (0.77)
- Grass Moving With The Win.M: 4.5-6.5 (0.75, pk -26.3, rms -46.1), 22-24 (0.72)
- 要"风过竹林"的床: **Wind 8 5-13** (wind in bamboo leaves 0.80) 或 Wind 7 3-6 (0.57)

---

## 3. 一次性音效: 过渡与点缀

标签集 (27 个): POS = delicate glass chime / soft wind chime / gentle harp glissando / soft bamboo wind chime / water droplet in a quiet pool / soft paper rustle / soft airy whoosh / small bell ringing softly / music box / kalimba note / gentle magical shimmer / bird wings flapping / leaves rustling / page turning / singing bowl / soft piano note / water splashing gently; NEG = harsh aggressive whoosh / cartoon sound effect / video game sound / distorted noise / sci-fi electronic sound / doorbell or alarm / church bell / coins or a cash register / glass breaking / explosion or impact。cheesy = cartoon + video game + distorted + sci-fi 之和。"take" 一栏是对整个 take 单独嵌入、用较小的专用标签集得到的分数。

### 3.1 钟、铃、钵 (推荐)

| 素材 | 干净窗口 (s) | 音高 (FFT 峰) | 移到 D 调 | pk / rms | CLAP 性格 | cheesy |
|---|---|---|---|---|---|---|
| **Bell Small** | **0.11-4.30** (-50 dB 尾在 5.13) | **2686 Hz = E7+32c** (纯音; 分音 7.07k, 13.2k) | E 是 D 大调 2 级: speed 0.9817 (去 32c); 或 0.8746 落到 D7 | -1.9 / -29.3 | take: delicate glass chime **0.82**; 2 s 窗里 doorbell 0.22, 宜轻、宜远 | 0.016 |
| **Bell Clang Resonance Low.M** (单声道) | **0.00-10.20** (前约 0.1 s 是敲击, 之后纯共鸣) | **656 Hz = E5-8c**, 另有 D6, B | 不用移 (D 五声吻合 0.99) | -10.2 / -30.2 | singing bowl **0.80** | 0.006 |
| **Bell Church 2** (5.1) | **0.86-7.61** (敲击在 0.86, 余音到约 19.5) | 基音嗡 **D4+17c (297 Hz)**, 敲击分音 G5-7c | speed 0.9902 (去 17c) | -5.5 / -24.4 | singing bowl 0.66, glass 0.19, church bell 0.09: 远寺钟感 | 0.008 |
| **Chimes 1** | **0.03-8.02** (1.0-2.0 一串, 之后稀疏叮当到 7.5) | 5.0-6.6 kHz 簇, 音级 C# 68% / B / G# | 在 D 大调内 (0.86, C# 为导音); 五声只 0.18 | -6.2 / -30.8 | take: glass chime **0.92**; 2 s 窗: music box 0.59 | 0.022 (video game 0.06) |
| Chimes 2 | 0.12-6.03 | E8 / A / A# / F 混合 | 五声 0.64 | -6.7 / -27.4 | glass chime 0.85 | video game 0.09 |
| Bell Cascade 2 | 0.11-5.12 | A / B / C# / C | 大调 0.73 (含一个 C 还原) | -5.6 / -26.6 | glass chime 0.87 | 0.05 |
| Bell Cascade 5 | 0.15-20.99 (取 3.5-8 最好) | C / G / D / F | **speed 0.8409 (降 3 半音) 后 D 五声 0.95** | -5.2 / -22.7 | magical shimmer 0.63 | 0.03 |
| **Shimmer Motion** | **0.09-10.09** (14-16 与 21.5-23.5 带 sci-fi, 避开) | **B5-31c (970 Hz)** + D6 | B 是 6 级: speed 1.018 (补 31c) | -7.4 / -25.1 | singing bowl 0.72-0.85, glass 0.12-0.16 | 0.057 |
| **MAGMisc_Magic Christmas Bells 2_344 Audio_Christmas** (单声道) | **0.80-17.52** | 2-16 kHz 随机细碎玻璃叮, 无调中心 | 不需要 | -17.1 / -39.1 (很轻) | take: glass chime **0.92**; 频谱是随机撒落, 没有雪橇铃节奏 | 0.011 |
| MAGShim_Shimmer Loop Small Bell Metal Taps_ESM_SGA3 | 15-20 或 37-39 | 7.5-10 kHz 金属轻敲, 无调 | 不需要 | 文件 pk -1.1; 窗 -5.3 / -21.9 | magical shimmer 0.40, music box 0.22 | 0.035-0.097 |
| Bells.M | 0.01-1.72; 2.44-4.34 | 第二下 G7-26c (3089 Hz) | G 在 D 大调内 | -6.5 / -33.0 | small bell 0.50, glass 0.22, doorbell 0.15 | 0.01 |
| Earring Jingling.M | 0.56-2.49 | 9.7 kHz | | -19.7 / -48.7 | small bell 0.42-0.54, doorbell 0.14-0.23: 极轻的金属闪光 | 0.017 |
| Drone Bells 01 (5.1) | 0.70-10.70 (可到 32.3) | **D3-32c** 持续音 | speed 1.0187 | -8.8 / -22.0 | singing bowl 0.79 | 0.022 |
| Appear Crystal 01 (5.1) | 0.58-10.58 | C#3+9c (+G4, F6) | speed 1.0595 (升 1 半音) | -11.4 / -20.3 | singing bowl 0.75 | 0.054 |
| Bell Clang Resonance High.M | 0.04-9.27 | 敲击 2688 Hz (与 Bell Small 同一分音) + 共鸣 **C5+42c** | 需 speed 1.0956 到 D5 | **0.0** / -29.4 (敲击满刻度) | singing bowl 0.58, glass 0.31, harsh metal clang 0.05 | 0.017 |
| UIMisc_Kalimba 3 Up_CB Sounddesign_APPlicable Sounds | 0.00-0.37 | F / A# / C | speed 0.9439 (降 1 半音) 后 D 五声 0.92 | -3.0 / -23.2 | music box 0.63, 偏 UI | 0.068 |

### 3.2 竖琴与琴弦扫

| 素材 | 干净窗口 (s) | 音高 | 移到 D 调 | pk / rms | CLAP | cheesy |
|---|---|---|---|---|---|---|
| **Harp Transition 1** | **0.08-7.77** (尾到 10.14) | C 大调上行 (C4 / E4 / G4; 音级 E 0.54, C 0.35, G 0.09) | **speed 1.1225 (升 2 半音) 后 D 大调吻合 0.99** | -7.7 / -28.0 | harp glissando 0.67 | 0.012 |
| **Harp Transition 2** | **0.06-5.45** (尾到 7.44) | F / Ab / Db / Bb (Db 五声) | **speed 1.0595 (升 1 半音) 后 D 五声 0.99** | -8.8 / -27.7 | harp glissando 0.76 | 0.019 |
| Piano Strings Transition 1 | 0.13-10.13 (尾到 16.1) | C3 / C4 / G#3 | 不太吻合 | -5.1 / -25.7 | harp 0.82, 但 soft/harsh 只有 0.17, 偏戏剧化 | 0.031 |
| Piano Strings Transition 2 | 0.04-10.04 | C5 / A#5 / D5 / D#5 | 不太吻合 | -5.0 / -27.9 | harp 0.73, shimmer 0.17 | 0.042 |

### 3.3 水滴

| 素材 | 窗口 (s) | CLAP | pk / rms | 音高 (水滴音高会上滑, 仅参考) |
|---|---|---|---|---|
| **Water Drip** | **20.50-22.50** | droplet in a quiet pool **0.85**, cheesy 0.04 | -13.4 / -46.9 | 约 980 Hz (B5-15c) |
| Water Drip 单滴 | 0.93-1.50; 5.40-5.85; 30.32-30.98 | single drop 0.77 / 0.66 / 0.63 | -14.7 / -8.9 / -14.8 | 约 853 Hz; 无明确; C6+27c |
| **Bathtub Drip** | **0.50-2.50**; 4.50-6.50; 10.50-12.50 | droplet **0.91-0.92**, cheesy ≤ 0.014 | -0.7 / -36.6; -5.7 / -41.6; -14.9 / -43.1 | 约 2.0 kHz |
| Bathtub Drip 单滴 | 19.90-20.22; 14.39-14.76 | single drop 0.86 / 0.79 | 很轻 (-31 / -36) | |
| Water Drips Drops.M | 0.31-2.58 | single drop 0.78 | -4.5 / -43.5 | F#7 附近 |
| Water Movement Light Splashes.M | 8.00-10.00 | droplet 0.62, splash 0.16 | -15.2 / -52.6 | cheesy 0.107 |

### 3.4 纸张

| 素材 | 窗口 (s) | CLAP (take, 纸张标签集) | pk / rms | 结论 |
|---|---|---|---|---|
| **A4 Printing Paper Rattle Page Turn Tail** (现用 PAPER) | **0.03-1.28** | soft paper rustle 0.49, page turning 0.39, plastic 0.07 | -10.5 / -41.5 | 保留 |
| **Paper Handle.M** | **3.41-3.97** | soft paper 0.60, page 0.28, plastic 0.07 | -9.4 / -39.0 | 推荐 |
| Paper Handle.M | 4.08-5.40 | soft 0.42, plastic 0.39 | -18.0 / -41.5 | 可用 |
| **Paper Movement on a Roll.M** | **1.44-1.87**, 0.62-1.08 | soft 0.64 / 0.55 | -22.8 / -53.4; -23.6 / -46.8 | 推荐, 轻 |
| Paper Movement on a Roll.M | 2.74-3.08 | soft 0.46, page 0.34, cartoon 0.16 | -18.2 / -45.1 | 尚可 |
| Large Book Page Turns.M | 26.04-26.59 | soft 0.59, page 0.32 | -30.9 / -57.7 | 轻柔翻页 |
| Large Book Page Turns.M | 22.00-23.68 | soft 0.40, page 0.30, plastic 0.25 | -4.0 / -35.0 | 大翻页 |
| Paper Handle Airline Tickets.M | 0.04-5.87 | soft 0.52, page 0.34 | -26.3 / -53.6 | 轻 |
| Map Handle 1.M | 0.26-1.77 | soft 0.56, page 0.22 | -7.4 / -40.8 | 可用 |

避开: Paper Handle.M 0.21-3.56 与 5.22-7.84 (cartoon 0.37-0.51); Paper Movement on a Roll.M 4.22-5.66 (plastic 0.72, cartoon 0.49); Book Pages Turning Flutter.M 与 PAPRMisc_Antique Books Slow Page Turns 6 (plastic crinkling 0.41-0.82 / 0.64, 偏塑料脆响, 不像宣纸); Encyclopedia Glossy Page Turn Muted (harsh whoosh 0.36-0.42); Tearing Paper Out.M (harsh whoosh 0.37, sci-fi 0.16)。

### 3.5 翅膀

| 素材 | 窗口 (s) | CLAP | pk / rms |
|---|---|---|---|
| **Wing Flaps Multiple Birds Fly Away** | **0.33-3.09** (第二阵 3.55-4.64) | wings **0.98**, cheesy 0.011 | -3.2 / -29.8 |
| **Bird Wings Flapping.M** | **0.17-4.53** (2.0-4.0 最佳) | wings 0.97 | -4.6 / -33.3 |
| **Eagle Wing Flaps.M** | **0.00-3.43** | wings 0.86, soft/harsh **0.65-0.73** (最柔的翅膀) | -12.7 / -39.6 |
| Bird Flying | 8.0-10.0; 整段 0.04-13.64 | wings 0.99 | -1.9 / -20.5 |
| Single Bird Flapping Wing.M | 0.28-2.60 | wings 0.85-0.92 | -7.3 / -32.3 |

### 3.6 Whoosh (soft vs harsh 二选一概率)

推荐 (柔):

| 素材 | 窗口 (s) | soft airy | soft/harsh 二选一 | pk / rms | 备注 |
|---|---|---|---|---|---|
| **Wind 4 阵风** | **48.30-50.70** | 0.43 | **0.90** | -5.2 / -19.1 | 天然阵风, 起伏 8.6 dB; 同类: 13.95-16.35 (0.79), 36.25-38.65 (0.78), 45.30-47.70 (0.77) |
| **Whoosh 09** | **0.22-2.53** | **0.51** (harsh 0.40) | 0.56 | -2.3 / -19.0 | 低频柔和过风, cheesy 0.016 |
| Whoosh 08 | 3.50-5.50; 6.00-8.50 | 0.26; 0.33-0.35 | 0.71; 0.64-0.68 | -1.8 / -17.0; -14.8 / -33.9 | 0-2 s 带一个 C#/A# 音调闪光 (shimmer 0.39, cheesy 0.15), 从 3.5 s 开始用 |
| Whoosh Down Flutter (5.1) | 10.00-13.50 | 0.34-0.42 | 0.51-0.58 | -24.0 / -37.5 | 7.5-9.5 偏翅膀 (0.47) |
| Eagle Wing Flaps.M | 0.00-3.43 | | 0.65-0.73 | -12.7 / -39.6 | 有机的翅膀风 |

**现用的 Whoosh 17 (harsh 0.84) 与 Whoosh 02 (harsh 0.91) 都偏硬。**

### 3.7 标记为廉价 / 卡通 / 科幻 / 失真 (不要用)

| 素材 | CLAP 证据 |
|---|---|
| MAGAngl_Magic Light Spell Enchantment Potion Effect Tonal Bright 03_ESM_FG2 | sci-fi 0.26-0.28, video game 0.19-0.26, cheesy 0.45-0.55 (0-3.5 s) |
| Liquid Drops FX 01 / 02 | video game 0.33-0.53, cheesy 0.44-0.74 |
| Interface Plucks Happy | video game 0.33, cheesy 0.52-0.57 |
| DSGNStngr_Power Up Bright Positive ... 05_ESM_AG | video game 0.36, sci-fi 0.32, cheesy 0.68 |
| GAMEMisc_Magic Creation 23 | cheesy 0.21-0.41 |
| Title Crystal Sweep 01 / 02 | video game 0.55-0.58 / harsh 0.39, cheesy 0.43-0.76 |
| Title Whoosh Flange 01, Title Digital Whoosh | video game 0.31-0.59, harsh 0.80-0.90 |
| Title Airy 01, Title Whoosh Tone 02 | harsh / explosion 0.16-0.22; whoosh tone 偏硬 |
| Whoosh 11 | sci-fi 0.41, video game 0.30, cheesy 0.77 |
| Whoosh 04, 12, 19 | cheesy 0.27-0.45 (19 含 cartoon 0.15-0.22) |
| Whoosh 01, 02, 03, 06, 07, 13, 14, 15, 16, 17, 18, Swish 1, Low Whoosh x2, Air Reverse Burst 1/2, WINDDsgn_Wind, Rush, Whoosh, Long x5 01 | harsh aggressive whoosh 0.57-0.96 |
| Swish 2, Woosh Sweep Slide Infographics Basic, Whooshes Quick Succession, Twirling Fast.M | cartoon 0.37-0.76 |
| Cofetti Whoosh Pluck Spill | sci-fi 0.42-0.48, cheesy 0.59-0.67 |
| Bell Cascade 1 (0-2 s) | video game 0.15, cartoon 0.15, cheesy 0.37 |
| Glassware Toast, Glass Clink Toast.M, Glass Cheers.M, GLASMvmt_Whoosh Glass Crystal Fragments | glass breaking 0.86-0.98 |
| Ting Coins | glass breaking 0.19-0.46, coins |
| Body Fall Leaves Grass.M | explosion/impact 0.25, glass 0.13-0.19 |
| Water Light Splashing.M | sci-fi 0.11-0.17, cheesy 0.23-0.30 (轻度) |
| Bell Church | 尾段 video game 0.11, cheesy 0.185 (轻度) |
| Clock Tower Chimes | CLAP 不算廉价 (music box 0.70), 但这是一段可辨认的钟楼报时旋律 (G6 / C7 / C6 / E7), 语义不对, 不建议 |
| UIMisc_Xylophone Ringtone 2 | E6, shimmer 0.33, 名字和听感都偏手机铃 |

---

## 4. 推荐床的底噪

床本身多是稳态噪声, 所以 floor10 约等于床本身; 真正要看的是低频哼/隆隆 (LF<120 Hz) 和高频嘶声 (>10 kHz)。hum 栏是 50/60 Hz 谱峰高出邻频多少 (≥ 20 dB 视为真有哼声)。

| 文件 / 窗口 (s) | rms | floor10 | LF<120 | 120 Hz-1 kHz | 4-10 kHz (底) | >10 kHz | 哼声 | 建议 |
|---|---|---|---|---|---|---|---|---|
| Forest Day 全段 | -29.1 | -42.4 | -60.9 | -47.8 | -45.8 (-51.4) | -63.0 | 无 | 最干净的鸟床; HPF 150 Hz 可选 |
| Forest Day 18-27 | -32.1 | -42.2 | -61.0 | -48.1 | -48.7 (-51.2) | -63.2 | 无 | |
| Forest Day 53-58 | -29.6 | -41.3 | -60.8 | -47.4 | -47.3 (-51.5) | -63.1 | 无 | |
| **Birds FX 02** 全段 | -20.3 | -26.3 | **-35.1** | **-25.3** | -48.4 (-50.4) | -58.3 | **50 Hz +38 dB; 148-235 Hz 音调** | **HPF ≥ 1.2 kHz** |
| Birds FX 02 0-3 | -17.0 | -25.6 | -33.6 | -25.4 | -25.5 (-48.2) | -53.4 | 同上 | 同上 |
| **Birds FX 01** 全段 | -15.7 | -19.2 | **-22.8** | **-20.8** | -39.8 (-42.1) | -45.3 | **49.9 Hz -24.7 dBFS** | **HPF 1.5-2 kHz** |
| Bird Sparrow 0-2.05 | -19.3 | -35.3 | -64.7 | -84.4 | -23.3 (-35.3) | -46.5 | 无 | 句间数字静音 |
| Country Day 2-10 (参考) | -39.7 | -41.8 | -63.5 | -48.4 | -47.3 (-49.6) | **-43.9** | 无 | 高频嘶声偏多 |
| Water Stream River 01 7-10 | -43.9 | -44.5 | -75.0 | -47.6 | -55.8 (-56.5) | -65.3 | 无 | 推高 10-15 dB 也干净 |
| Forest Stream River Water 01 17-25 | -38.0 | -38.8 | -84.0 | -41.6 | -51.8 (-52.7) | -63.2 | 无 | |
| Water River 2 15-23 | -30.2 | -32.2 | -76.8 | -36.0 | -39.8 (-41.5) | -46.2 | 120 Hz +18 dB (带电平 -76, 听不到) | |
| Water Lake 1 19-23 | -28.7 | -37.0 | -67.0 | -35.4 | -47.7 (-52.9) | -56.3 | 无 | 拍岸浪起伏 3.4 dB |
| WATRLap ... 10-18 | -33.5 | -36.9 | **-53.1** | -38.1 | -44.4 (-47.0) | -53.3 | 无 | HPF 80-100 Hz |
| WATRLap ... 37-40 | -39.5 | -41.6 | -49.1 | -45.0 | -47.4 (-52.7) | -57.8 | 120 Hz +18.8 | HPF 100 Hz |
| Wind 5 23-31 | -19.4 | -21.3 | -30.2 | -21.2 | -36.9 (-38.3) | -46.6 | 无 | HPF 120 Hz |
| Wind 5 33-41 | -18.8 | -20.3 | -29.6 | -20.2 | -37.2 (-38.4) | -46.2 | 无 | HPF 120 Hz |
| Wind 6 29-35 | -19.0 | -20.4 | -45.3 | -22.0 | -36.5 (-38.1) | -43.9 | 无 | 通透 |
| Wind 8 1-7 | -34.1 | -36.1 | -42.8 | -38.0 | -48.9 (-53.7) | -49.5 | **60 Hz -50 dBFS (突出 66 dB)** | HPF 100 Hz |
| Wind 8 35-43 | -29.9 | -33.5 | -40.9 | -34.7 | -46.1 (-48.2) | -46.8 | 60 Hz +17.6 | HPF 100 Hz |
| Wind 3 10-13 | -25.9 | -27.3 | -43.6 | -27.1 | -47.2 (-48.7) | -59.3 | 无 | |
| Wind 7 4-9 | -33.3 | -34.5 | -37.6 | -38.5 | -46.2 (-46.6) | -50.2 | **60 Hz -47.7 dBFS (突出 74 dB) + 120 Hz** | HPF 150 Hz |
| Wind 4 7-15 (现用) | -18.8 | -21.8 | -40.4 | -19.8 | **-69.7** | -76.0 | 无 | 频带受限, 闷 |

各文件头尾渐入渐出 (film_time 会绕回文件开头, 绕回点会掉坑): Water Stream River 01 与 Forest Stream 头 1.0 s、尾约 5 s (+ 2.3-2.5 s 数字静音); Wind 3/5/6/7/8 头 0.5-1.0 s、尾 1.5-2.5 s; Forest Day 头 1.5 s、尾约 2 s; Country Day 头 1.0 s、尾约 2 s; Birds FX 02 无渐变。

---

## 5. 给混音的建议

1. **删除 Bird Seagull 的全部三处用法** (33.8, 70.2, 双耳 orb 48.0)。替换建议: 33.8 "spring on the shore" 用 Birds FX 02 0.00-2.80; 70.2 "far birds in the morning forest" 用 Birds FX 01 0.00-1.45 或 Bird Sparrow 0.00-2.05; 48.0 环绕鸟用 Birds FX 02 14.70-17.20 / 23.50-26.00 / 47.90-49.90 (都要 HPF 1.2 kHz)。
2. **换掉 Country Day 的四张床** (27.4-34.2, 48.3-52.54, 73.0-80.0, 80.0-87.8)。59.6-66.8 那张不含乌鸦, 但 60.5-62.3 有接缝掉坑。可用: Forest Day 干净区段 (用 `a=` 对位) + 一层通透的风 (Wind 5 33-41 或 Wind 8) + 稀疏的 Birds FX 02 歌唱段。
3. **"一水" 57.43-59.6** 换成 Water Stream River 01 7-10 (最像小溪), 或者仍用 Forest Stream 但让读取落在 17-25 s。
4. **竹林 42.6-48.3** 用 Wind 8 5-13 (竹叶风) 代替 Wind 4。
5. **Whoosh 17 (57.2, 93.55)** 换成 Wind 4 阵风 48.30-50.70 或 Whoosh 09 0.22-2.53; 93.55 "托举" 也可叠 Wing Flaps Multiple Birds Fly Away 0.33-3.09 或 Eagle Wing Flaps.M。
6. 片名与转场的点缀可从 3.1 / 3.2 节选, 统一到 D 调: Harp Transition 1 (speed 1.1225), Harp Transition 2 (speed 1.0595), Bell Clang Resonance Low.M (不用移), Bell Small (speed 0.9817), Shimmer Motion (speed 1.018), Bell Church 2 (speed 0.9902)。
7. 融合: 所有点缀鸟先 HPF, 再用同一个房间/混响送到床的距离感里; 不要让 Forest Day 的同一段 (每约 11 s 重复的下行颤音) 在片中多次出现。
