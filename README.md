# 使用人體骨架關鍵點對健身影片進行動作分類

## 一、簡介
本專案將有深蹲、臥推和硬舉等動作的影片進行分類，跳脫直接使用 CNN 進行圖片動作辨
別的方法，使用 Mediapipe（人體關鍵點識別 pre-trained model）將人體骨架 32 個關鍵點抓出
後，再以人體關節角度、或是人體關鍵點比例分佈使用監督式模型 LSTM、Random Forest，
非監督式模型 K-means Clustering 等模型進行訓練與分析，來達到為分類影片動作的效果。
## 二、資料集
### 二-1、影片來源和篩選
考量到資料集要符合標籤內容，選擇使用 Youtube Shorts 上面的深
蹲、臥推和硬舉影片，以 FreeMake下載，影片內容以多 reps（一組
多下）、秒數較短、無其他多餘動作影片為主。
因為多 reps 的影片中每部影片能採樣到較多重複動作，且短秒數的
影片不會有太多無關動作，如：暖身、錯誤示範、講解動作，且動
作通常以一組為主，不會有其他多餘休息時間，導致另外花費時間
清洗資料。另外除了以單一角度進行拍攝的影片，盡可能囊括從正
面、背面、側面、側後、側前拍攝的影片。
### 二-2、片段
蒐集影片後，將影片分割成小片段，為模型訓練將每段片段的幀數設置一樣，有 70, 100,
120, 150 幀等四種模式的片段長度，並將每個模式存取為三個.npy檔（label, angle, point）。考
量到影片開頭通常為預備動作或是暖身，與實際期望採樣動作有偏差，如：臥推前坐在臥推
座上、深蹲前的背槓預備動作、硬舉前的深吸氣準備等，通常忽略在影片中的第一個片段。
而影片中最後一段片段若是未達到要求幀數，則會捨棄該片段。
### 二-3、使用 Mediapipe 提取每幀特徵
將切割好的片段使用 Mediapipe 對每一幀進行骨架關鍵點提取，提
取出的關鍵點有 33 個，每個關鍵點有（x, y, z, 信心程度）四個特
徵，此處的 x, y, z 為關鍵點在此幀中的比例相對位置，介於 0 ~ 1
之間，並非在此幀中的絕對長寬高。
利用這些關鍵點，算出左右邊的肩膀、手肘、膝蓋共六個身體關節
角度，每個角度需要三個點的三維資料進行運算，並保證關節角度
透過 +360 度以保證總是大於 0，避免模型無法識別一樣的角度。
並將該資料標注為影片動作。

兩張圖片來源：https://developers.google.com/mediapipe

### 二-4、儲存資料集
共 46 部影片，有 4 種片度長度，1 個片段存為 3 個種
類如：{video}_{sliceId}_{type}.npy 。並將.npy 存取在
data{sliceLength}\{motion}\{type} 資料夾之下。再將三組
資料集各自拆分為訓練集 80%、驗證集 10%、測試集
10% 進行後續操作。

## 三、演算法與模型
### 三-1、監督式學習 Long Short-Term Memory
#### 三-1-a、LSTM 說明：
因考量影片有動作連續的性質，所以使用 LSTM 模型進行訓練。此部分僅使用關鍵點進行學
習，不使用關節角度。模型架構使用了 12 層的 LSTM，第一層使用 256 個神經元，對應輸
入的關鍵點數據。之後的 11 層 LSTM 各使用 16 個神經元，用於提取時序特徵，在測試時發
現當深度較深時，訓練集的準確率才有穩定上升的效果，神經元數目之廣度倒是沒有太大差
別。而最後一層 LSTM 使用 32 個神經元，將提取到的特徵進行整合。
library 為 Keras

Labels.npy (sliceLength, labels = 3)
Angles.npy (sliceLength, angles = 6)
Points.npy (sliceLength, position = 4(x, y, z, valid) * 33(keypoints))
接著經過 3 層的全連接層 (Dense)，分別使用 64、32 和 3 個神經元。其中最後一層因為是分
類問題，使用 softmax 激活函數，將特徵 output 到 3 種結果，代表 3 種不同的動作。
在訓練過程中，學習率原本設為 0.001，但發生梯度爆炸問題，因此下修到 0.0001 後梯度穩
定下來（會依照幀數、模型深度、模型神經元數目有關）。Batch size 設為 20，訓練 200 個
epoch。使用 Adam optimizer 來優化模型，loss function 採用 categorical cross-entropy。訓練過程
中同時在驗證集上進行評估，並使用 TensorBoard callback 記錄訓練過程以方便觀察模型訓練
狀況。
#### 三-1-b、LSTM 結果分析（以 70 frames 為片段長度說明）
Test: Accuracy: 0.8052, Test Loss: 0.4708 0: 硬舉 1: 深蹲 2: 臥推
LSTM 在深蹲和臥推上表現較好，精確率、召回率和 F1 score 都算高。但在硬舉上的表現相
對較差，有將硬舉誤判為深蹲的情況，判斷原因是可能硬舉前的準備動作為站立姿勢，且準
備時間較長，可能會與深蹲的背槓姿勢混淆導致誤判。而 ROC 曲線和 AUC 的結果顯示，整
體上具有足夠的分類能力，在不同的閾值下都能夠平衡 TPR 和 FPR。
### 三-2、監督式學習 Random Forest
三-2-a、Random Forest 說明：
與 LSTM 不同的地方在於，這部份我認為比起使用關鍵點座標，用關節角度（angles.npy）來
訓練較好，因為關節角度能更簡單地表示人體姿態，如：深蹲肘部彎曲就與硬舉的手臂伸直
有很大的區別，但若使用關鍵點就較難看出。另外若是使用關鍵點座標，可能會因為拍攝角
度的關係混淆模型，或是因為模型可能因為無法使用深度模型，而無法做出特徵抽取，導致
訓練效果很差。而在 Random Forest 的參數設置上，這邊選擇了 100 棵決策樹。
3
Random Forest 結果分析（以 70 frames 為片段長度說明）：
Library 為 Scikit-Learn
Test Accuracy: 0.8831 0: 硬舉 1: 深蹲 2: 臥推
隨機森林模型在深蹲和硬舉動作的分辨上都較 LSTM 優秀，猜想是應該剛剛提到的問題（硬
舉跟深蹲的準備動作相像）所導致。得以看出確實使用關節角度可以較好的分辨硬舉和深蹲
的差距，雖然在深蹲上還是有些許誤判的情況，但在硬舉上有很高的精確率、召回率和 F1
score。ROC 曲線和 AUC 的結果進一步驗證了模型的分類能力，在不同的閾值下都能很好地
平衡 TPR 和 FPR。測試集上的準確率達到比 LSTM 還要好的 88.31%。
### 三-3、非監督式學習 K-means
使用 K-means 演算法對關節角度，而非關鍵點的資料集數據進行 Clustering，設 K = 3。為了
評估 Clustering 的狀況，使用輪廓係數 (Silhouette Score) 來衡量聚類的緊湊性和分離性。輪廓
係數的範圍介於 [-1, 1]，值越高表示聚類效果越好。在這個案例中，經過多次實驗輪廓係數
通常在 0.2 之下，準確率則是在 0.3~0.5 之間，成效沒有到太好。
另外在更新的版本，我對數據進行了一些預處理，包含特徵化、降維等等。但輪廓係數則是
同樣在 0.2 之下，準確率則是在 0.45~0.65 有些許提高。
4
Library 為 Scikit-Learn
版本一 版本二
## 四、實驗與分析
### 四-1、資料集取材方向
而在 LSTM 學習初期，一開始使用的不是非短秒數、多 reps 的影片，而是使用教學講解影片
，並且沒有特別剪輯，但訓練出的效果非常糟糕，準確率不會穩定上升，震盪很大，且通常
介於 0.2 到 0.4 間，且怎麼訓練都起不來。於是後面重新取材，選擇以動作為主，蒐集各拍
攝角度，不會穿插其他畫面的影片為主。
未適當篩選前分類準確率- 800 epochs 經適當篩選前分類準確率-200 epochs
### 四-2、片段長度的影響
片段長度對於 LSTM 的影響比起 Random Forest 和 K-means 大上許多，原因推測可能是 LSTM
是由每個片段下去做學習，而 Random Forest 和 K-means 的使用計算角度的資料量是依總幀
數，所以在片段長度不同的情況只會有些許差異，如：頭尾兩個片段的捨去部份。僅有微小
差異，其他部份則都是使用差不多數據下去試驗。
而在片段長度則是發現，當片段長度越短時，LSTM 的反應才會比較好，以下是用同樣參數
訓練模型（學習率: 0.0001, Optimizer: Adam, Batch Size: 20 Epoch: 200）從 70, 100, 120, 150 所得
出的 testing set 準確率結果。
其中一旦長度超過 120 frames 就會容易發生梯度爆炸的情況發生。當然，上面結果是使用同
樣參數所得到的，但實際上各片段長度的模型是要個別再去 fine-tune 的。如使用較廣的模型
所訓練的 150 frames 也可得到 Test Loss: 0.9179 Test Accuracy: 0.6875 的結果。
不過就結論來看，當片段長度拉長後，模型較不容易分別出深蹲跟硬舉的差別，而當使用片
段長度較短的資料集來訓練，切割會更精細，得到結果更好。

## 五、其他
### 五-1、MoveNet, OpenPose, MediaPipe?
對於這次的專案模型原本是使用 Tensorflow 的 MoveNet 來做骨架特徵提取的，而 MoveNet 雖
然較快速，但是只有二維資料，且人體關鍵點的數量沒有 Mediapipe 這麼多。同樣 OpenPose
的效果也跟 MoveNet 差不多，一樣是二維的，且關鍵點沒那麼多。所以選擇用 Mediapipe。
### 五-2、更好的骨架辨別模型
原本是想要用 Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action
Recognition (ST-GCN) 來作為這次監督式的深度模型，但是 ST-GCN 需要用到他們的資料集，
若是要用自己搜集的資料集需要剪輯非常多的影片，並且改 ST-GCN 的 configuration 還要從
MediaPipe 再改成 OpenPose 才有辦法使用，而在這份專案中最麻煩的部份就是是在處理資料
的部份，一旦要改的話又需要花很多時間寫處理資料的 script ，因此在一頓研究之後，還是
選擇使用 LSTM 配合 MediaPipe 來做人體動作辨別。
### 五-3、片段長度
前面花了很大的篇幅在講片段長度對於模型訓練的效果，最後決定的片段長度其實除了模型
訓練出來的效果外，也參考實際動作一遍粗略估計的幀數，同樣也參考別人做 LSTM 做影片
識別的幀數，因為原本我以為當幀數越大，效果會越好時，完全搞錯方向所以訓練不出來感
到很慌，最後也推測其實跟片段數目（片段長度越短，一個影片的片段數目也會越多）也很
有關係。
### 五-4、模型深度
在訓練的過程中，嘗試過將模型（Dense, LSTM）加深，或是在同一層中加入更多神經元，
發現加入更多神經元，較容易有梯度爆炸的情況產生。另一方面當我嘗試將 LSTM 層加深，
效果會明顯變好，所以就一直疊，最後 LSTM 層變得超深。考慮到當 70 幀時，準確率最後
快到 100%，為避免 Overfitting 有加入 Dropout 層來讓結果更好。
### 五-4、未來展望
ST-GCN
辨別影片動作的方式有很多種，而這次因為要做非監督式和監督式等三種模型，其中只有一
種是深度模型，因此特別想了用人體骨架的方式實現。而其中我對 ST-GCN 非常有興趣，若
是不用自行準備資料集，可以用看看 ST-GCN 提供的模型嘗試來對動作識別，或是在蒐集更
多影片，剪輯成同一格式，並用 PyTorch 重現 ST-GCN 處理 Mediapipe 所得到的三維資料。
Real-Time
另外，可以加入 Real-Time 的功能，提供攝影機即時辨別動作，並連動作骨架一起繪製在旁
邊的功能，甚至可以增加更多動作類別來訓練模型，如：引體向上、羅馬尼亞深蹲、舉重
等。
網站或 APP
可以配合 Flask 架設程網站或 APP，提供使用者在線判斷動作等等功能。

## 六、參考資料
### 六-1、Reference
https://blog.csdn.net/qq_42599237/article/details/111566607 
https://blog.csdn.net/weixin_48159023/article/details/121604637
https://tengyuanchang.medium.com/%E6%B7%BA%E8%AB%87%E9%81%9E%E6%AD%B8%E7%A
5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF-rnn-%E8%88%87%E9%95%B7%E7%9F%AD%E6%9
C%9F%E8%A8%98%E6%86%B6%E6%A8%A1%E5%9E%8B-lstm-300cbe5efcc3
https://bleedaiacademy.com/introduction-to-pose-detection-and-basic-pose-classification/
https://www.youtube.com/watch?v=xCGidAeyS4M
https://www.youtube.com/watch?v=doDUihpj6ro&t=6467s
https://www.youtube.com/watch?v=rTqmWlnwz_0&t=44s
https://ithelp.ithome.com.tw/m/articles/10305134
https://mediapipe-studio.webapps.google.com/home
https://ithelp.ithome.com.tw/m/articles/10322270
### 六-2、Dataset - Youtube Shorts Videos
臥推  
├── 120kg bench press 20 reps for 3 sets easy.mp4  
├── 120kg x 20 reps Bench Press.mp4  
├── 125kg_275lb bench press 5 rep PR.mp4  
├── 185lbs bench press for 20 reps.mp4  
├── 225 Bench press for 10 reps.mp4  
├── 315 bench press for a million reps.mp4  
├── 315 x 20 bench (new pr).mp4  
├── 325 bench 20 reps RAW @ 19yr 250-255.mp4  
├── 335 Bench Press for 5 rep's.mp4  
├── 585 BENCH FOR 23 REPS!.mp4  
├── 60 KG Bench Press 30 reps 15 years old.mp4  
├── 60KG Bench Press for 9 Reps (10th rep fail).mp4  
├── Allen Baria - Bench Press 405 lbs X 25 reps 500 lbs X 10 reps.mp4  
├── BENCH PRESS 225 LBS _100 KG_ My first time ever.mp4  
├── Bench Press 315 lbs 10 reps (body weight 205 lbs) BHS.mp4  
├── 卧推｜ .mp4  
└── 臥推80公斤3下.mp4

硬舉  
├── 150kg deadlift - 5 reps.mp4  
├── 150KG DEADLIFT TRIPLE PR!.mp4  
├── 180 KG Deadlift 10 Reps by a 15 year old.mp4  
├── 200 KG DEADLIFT @ 73 KG BODYWEIGHT.mp4  
├── 200kg deadlift at 17 years old (70kg bw).mp4  
├── 200 kg Deadlift at 70 kg Bodyweight.mp4  
├── 220kg deadlift for 20 reps.mp4  
├── 700x10 Deadlift.mp4  
├── Bradley Martyn _405lb deadlift for 20_ RAW.mp4  
├── COULD YOU MANAGE THIS 300KG_661LBS X 10 REPS DEADLIFT.mp4  
├── Deadlift 150 KG 🏋 Convenational lift .mp4  
├── Deadlift 150 kg.mp4  
├── Deadlift 200kg_440lbs .mp4  
├── Deadlift 250kg 551 lbs for 10 Reps 30_10_20.mp4  
├── Derek Poundstone 750lb Deadlift for 10 reps.mp4  
└── 健身五個月，.mp4

深蹲  
├── 100 KG squat 8 reps RAW.mp4  
├── 100KG Squat for 9 Reps Beltless .mp4  
├── 140 KG Squat 10 reps RPE 5.mp4  
├── 140kg squat 4 reps PR! - Road to a 4 plate squat.mp4  
├── 150kg_330lbs squat for 5 reps.mp4  
├── 180kg Squat x 10reps.mp4  
├── 700lbs Squat. No belt or sleeves..mp4  
├── Back Squat 2x Body Weight 10 reps.mp4  
├── Frederick Luethcke - Back Squat - 151kg for 15 Reps.mp4  
├── RAW Squat 300kg x 6 reps _ 660 pounds.mp4  
├── Ray Williams (800 lbs) 5 reps.mp4  
├── Squat 100kg x 30 (20 rep squats and milk program).mp4  
└── 深蹲 Squat 270公斤.mp4
