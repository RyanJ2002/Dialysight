# DialySight 使用教學

這份文件只講「怎麼操作」。設計理由見 [README](../README.md) 與 [architecture.md](architecture.md)。

```
床邊主機（相機）                    院內網路                        護理站電腦
run_demo.py                     只有文字，沒有影像                nurse_station.py
影像只在這裡處理  ── 事件 + 每 5 秒心跳 ──▶  /ingest  ──▶  瀏覽器開 http://localhost:8420
```

---

## 0. 第一次使用（只做一次）

```bash
pip install -r requirements.txt
python scripts/selftest.py        # 55 項全過 = 環境沒問題
```

> Windows 終端機出現 `UnicodeEncodeError` 是終端機編碼問題，不是程式壞了。
> PowerShell 先執行 `$env:PYTHONIOENCODING="utf-8"` 再跑一次。

---

## 1. 選一種跑法

| 我想… | 怎麼做 |
|---|---|
| **A. 只看護理站畫面**（沒有影片、沒有鏡頭） | `python scripts/nurse_station.py --demo`，瀏覽器開 http://localhost:8420 |
| **B. 床邊 + 護理站在同一台電腦**（demo 最常用） | 見下方，**不用改任何設定** |
| **C. 床邊和護理站在不同電腦** | 見下方，要改床邊 config 的一行 |

### B. 同一台電腦

開兩個終端機：

```bash
# 終端機 1：護理站
python scripts/nurse_station.py

# 終端機 2：床邊（用錄好的影片）
python scripts/run_demo.py --video data/clips/clip_03_gradual_drift.mp4 --show
```

瀏覽器開 http://localhost:8420，Bed 07 會出現在床位總覽。影片播完時，床邊會送出「治療結束」，
「紀錄整理」頁就會多一份草稿。

> config 預設就送到 `http://localhost:8420/ingest`。沒開護理站也沒關係，床邊照常偵測，
> 事件一樣寫進本機的 `results/events_Bed07.jsonl`。

### C. 不同電腦

1. 在**護理站電腦**啟動 `python scripts/nurse_station.py`。它會印出這台電腦的網址，例如：
   ```
   1. 用瀏覽器打開      http://localhost:8420
      同網路的其他電腦  http://192.168.1.23:8420
   2. 床邊主機要把資料送到這裡：
      別台電腦    床邊 config 的 sink.http 改成 http://192.168.1.23:8420/ingest
   ```
   不知道本機 IP 也沒關係，照它印的抄就好。有多個 IP 時，選跟床邊同一個網段的（通常是 `192.168.` 開頭）。
   也可以自己查：PowerShell 執行 `ipconfig`，看「IPv4 位址」。
2. 在**床邊電腦**打開 `config/default.yaml`，把 `sink.http` 那一行的 `localhost` 換成上面的 IP。
3. 床邊照常執行 `run_demo.py`。
4. 連不上時，先確認兩台在同一個網路，以及護理站電腦的 Windows 防火牆允許 Python 使用 8420 port。

---

## 2. 床邊畫面

```bash
python scripts/run_demo.py --video data/clips/clip_03_gradual_drift.mp4 --show    # 即時視窗
python scripts/run_demo.py --video data/clips/clip_03_gradual_drift.mp4 --out results/demo.mp4   # 輸出影片
python scripts/run_demo.py --video 0 --config config/webcam.yaml --show           # 筆電鏡頭
```

視窗裡按 **L** 切換中文／英文，按 **Q** 結束。

### 狀態怎麼看

| 畫面上 | 意思 | 會通知護理站嗎 |
|---|---|---|
| 建立基準 | 前 30 秒學這位病人的靜止姿態，病人請保持不動 | 否 |
| NORMAL（綠） | 手臂在基準範圍內 | 否 |
| WATCHING（黃） | 偏離超過 0.15，正在確認，進度條跑滿 5 秒才升級 | 否，只記錄 |
| ALERT（紅） | 偏離持續 5 秒以上 | **是** |
| 冷卻 | 回到 0.10 以下 3 秒後解除，之後 30 秒不重複響 | 否 |
| 無法判定（灰、斜線） | 看不到穿刺側手臂。**這不是「正常」，是「看不到」** | 否，但護理站會標出來 |

另有**快通道**：突然坐起、手臂大幅移動，不到 1 秒就通知，不受上面的流程影響。

### 各區塊

- **影像**：灰色虛線是治療開始時的手臂位置，彩色實線是現在的位置，旁邊標著偏離量。
- **狀態卡**：說明為什麼還沒通知、或已經通知多久。
- **趨勢圖**：最近 90 秒的偏離量，上面有進入與退出兩條門檻線；正在上升時會預估約幾秒後越過門檻。
- **訊號投票**、**雙通道**：各訊號目前的分數，以及慢通道、快通道的狀態。
- **底部**：整段治療的時間軸（色帶就是狀態，刻度是事件），以及裝置、FPS、CPU。

---

## 3. 護理站

瀏覽器開 http://localhost:8420。**第一次使用**：
1. 左下角點名字，換成你的名字（確認告警、簽核時會記下來）。
2. 點左下角「提示音」打開聲音。瀏覽器規定網頁要先被點一下才能出聲。

左側選單有四頁：

| 頁面 | 看什麼 | 你要做的事 |
|---|---|---|
| **床位總覽** | 每床一張卡片，顏色就是狀態；點卡片，右側看詳情（最近 15 分鐘趨勢、本次治療時間軸、這床的事件） | 紅色卡片按 **確認處理**；要到床邊處理按 **照護中**（暫停通知 10 分鐘），處理完可按 **提前恢復** |
| **事件與告警** | 這一班所有事件：告警、觀察、無法判定、照護中，可以篩選 | 未確認的告警整列淡紅，可以直接按 **確認** |
| **紀錄整理** | 治療結束的床次與自動產生的草稿 | 左邊選一床 → 看草稿 → 填「護理師補充」→ 勾第 2 項 → **簽核並寫入** |
| **報表與裝置** | 本班統計、各床連線狀態、院內網路上實際傳了什麼 | 只看，不用操作 |

- 告警超過 60 秒沒人確認，卡片會閃，提示音每 20 秒再響一次。
- 照護中的那 10 分鐘，床邊仍然照常偵測與記錄，只是不通知。
- 確認、照護中、簽核**只存在護理站**，不會回送床邊。
- 紀錄整理時可以按「編輯草稿」修改。改到事件紀錄裡沒有的數字（例如把 42 秒改成 45 秒），右側第 1 項會變紅並列出來；
  仍要簽核時會再問一次，核對結果會跟紀錄一起寫入。

---

## 4. 接 UGen300 上的 LLM（選用）

不接也能用：紀錄草稿預設是**範本填空**，功能完整。接上之後，LLM 只負責把範本的句子改寫得更通順。

### 用的是什麼

Hailo 官方的 **hailo-ollama**（[hailo_model_zoo_genai](https://github.com/hailo-ai/hailo_model_zoo_genai)）：
它在 Hailo-10H 上跑 LLM，對外提供跟 Ollama 相容的 HTTP API，預設 port 8000。
你說的「有簡單 UI 的 app」應該是 **Open WebUI**：它是給人用的聊天介面，背後連的也是 hailo-ollama。
護理站直接連 hailo-ollama 的 API，不經過 Open WebUI。兩個可以同時開著，Open WebUI 拿來自己試模型很方便。

### 只有一張 UGen300 怎麼辦

- **現在**：姿態偵測跑在 CPU（`config/default.yaml` 的 `inference.backend: ultralytics`），UGen300 整張空著，可以全部給 LLM 用。
- **之後姿態偵測改用 Hailo 時**：LLM 只在治療結束、起草紀錄時跑一下，不需要一直佔用。
  hailo-ollama 的文件寫到，其他程式可以用同一個 VDevice group id 跟它共用裝置
  （環境變數 `HAILO_OLLAMA_VDEVICE_GROUP_ID`）。這一點我們還沒實機測過。
- hailo-ollama 一次只處理一個請求、同時只能載入一個 LLM。護理站會自己排隊；
  如果剛好有人在 Open WebUI 聊天，草稿可能逾時，這時會自動退回範本，不會卡住。

### 步驟

1. 照 [hailo_model_zoo_genai](https://github.com/hailo-ai/hailo_model_zoo_genai) 安裝 HailoRT 與 hailo-ollama。
   官方主要支援 Ubuntu（`.deb` 或 Docker）；Windows 要自己從原始碼編譯。
2. 啟動：`hailo-ollama`
3. 下載模型（只做一次）：
   ```bash
   curl http://localhost:8000/hailo/v1/list          # 看有哪些模型可以下載
   curl http://localhost:8000/api/pull -H "Content-Type: application/json" -d '{"model": "qwen2:1.5b", "stream": true}'
   ```
4. 啟動護理站時加上 `--llm`：
   ```bash
   python scripts/nurse_station.py --llm http://localhost:8000
   ```
   啟動畫面第 3 行會顯示「已連上 hailo-ollama，模型：…」。hailo-ollama 在別台電腦就把 localhost 換成它的 IP。
   預設自動挑已下載的模型（Qwen 優先），要指定就加 `--llm-model qwen2:1.5b`。
5. （選用）Open WebUI：
   ```bash
   docker run -d --net=host -e OLLAMA_BASE_URL=http://127.0.0.1:8000 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
   ```
   瀏覽器開 http://localhost:8080。

### 安全機制

- 送給 LLM 的只有範本整理好的事件文字，沒有影像、沒有座標。
- LLM 的輸出只要有一個數字對不上事件紀錄、用了沒辦法核對的中文數字、或出現「建議」這類臨床建議用語，
  整份就退回範本。退回的原因會顯示在紀錄整理頁右側。
- 簽核的紀錄會記下這份草稿是範本產生，還是哪一個模型改寫的。

---

## 5. 其他常用指令

```bash
# 左右對照影片：單一門檻 vs DialySight
python scripts/run_comparison.py --video data/clips/clip_03_gradual_drift.mp4 --out results/comparison.mp4

# 輸出一份事件紀錄（markdown）
python scripts/run_demo.py --video data/clips/clip_03_gradual_drift.mp4 --report results/clip_03_session.md

# 床邊不寫事件檔、也不送護理站
python scripts/run_demo.py --video data/clips/clip_03_gradual_drift.mp4 --show --no-sink
```

門檻、確認時間、冷卻時間在 `config/default.yaml` 的 `state_machine` 區塊。
床邊畫面預設中文；要預設英文，在 `visualize` 區塊加一行 `language: en`。

---

## 6. 檔案在哪裡

| 檔案 | 內容 |
|---|---|
| `results/events_Bed07.jsonl` | 床邊送出的每一筆事件與心跳（裡面沒有任何影像） |
| `results/nurse_station.jsonl` | 護理站收到的全部內容，加上確認、照護中的操作紀錄 |
| `results/nursing_records.jsonl` | 護理師簽核後的護理紀錄 |
| `results/*_demo.jsonl` | `--demo` 模式寫的檔案，跟正式紀錄分開 |

---

## 7. 常見問題

| 狀況 | 怎麼辦 |
|---|---|
| 瀏覽器打不開 http://localhost:8420 | 護理站沒在跑，或啟動時顯示 port 被占用：加 `--port 8421`（床邊 config 的網址也要跟著改） |
| 床邊跑了，護理站卻沒出現那一床 | 看 `run_demo.py` 結束時印的 `"sink": "http"` 區塊：`failed` 很多就是連不上，檢查網址、防火牆 |
| 告警時沒有聲音 | 左下角「提示音」要先點一次打開 |
| 紀錄整理是空的 | 床次要「結束」才會出現。影片播完或按 Q 結束床邊程式時，會送出「治療結束」訊號 |
| 想讓床邊畫面顯示英文 | 執行中按 L，或在 config 的 `visualize` 區塊加 `language: en` |
