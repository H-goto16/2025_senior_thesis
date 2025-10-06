## 論文の趣旨
- 一般ユーザでも「自分用の画像認識モデル」を収集・ラベリング・学習・評価・運用まで一気通貫で回せるアプリケーションの構築。
- 開放語彙検出（YOLO-Worldの思想）を取り入れ、語彙（クラス）をユーザ側で柔軟に追加しつつ、推論を高速・軽量に維持する運用を重視。
- UI（React Native + Expo）とAPI（FastAPI）を統合し、非専門家が繰り返し改善できる実務的フローを提供。

## システム構成（把握）
- フロントエンド: React Native + Expo、Zustand、Expo Router、Plotly描画
  - 主要タブ: Detection / Labeling / Training / Models / Analytics
  - 役割: 画像取得、ドラッグ＆ドロップでのラベリング、学習トリガ、進捗／履歴／指標の可視化、モデルの切替・検証
- バックエンド: FastAPI
  - 主要API: /model/classes, /detect, /labeling/submit, /training/start(-async), /training/history, /training/metrics/{run}, /models/*
  - 学習・推論: Ultralytics YOLO を中核とし、学習結果の自動ロード、モデル一覧・切替・検証に対応
  - データ分析: クラス分布や画像サイズ分布、BBox統計などのAPI

## 実装上のポイント（把握）
- 語彙管理: APIから登録→永続化（custom_vocab.json）→再起動後復元
- 推論運用: prompt-then-detectパラダイムを意識（事前語彙化で実行時に言語エンコーダ不要）
- 学習運用: 同期/非同期の両エンドポイント、出力best.ptの自動ロード、進捗取得API
- 可視化: 学習履歴（results.csv, args.yaml）からPlotlyで可視化、比較APIあり

## 実験（把握）
- 領域検出精度（4章）
  - 目的: セグメンテーション（mIoU）と物体検出（mAP/Precision/Recall）を比較・検討
  - データ: 241枚、YOLO形式ラベル、train/val分割（現状は簡便化）
  - 指標: IoU/mIoU、AP/mAP、YOLOX Loss（Box/Objectness/Classification）
  - 結果の表: Backbone×Seg手法×YOLOXの組合せ比較（FLOPs/Params/Latency含む）
  - 図の参照は残し、画像は一旦コメントアウト済（後で差し替え）

- 残量測定（5章）
  - 目的: 推定精度（RMSE）を評価
  - データ: 34枚（各画像の実測[%]付き）
  - 手法: 球冠ベース / 関数ベース（食器形状近似）
  - 指標: RMSE、mIoU（Seg/Obj）との関係を考察
  - 図は参照維持、画像は一旦コメントアウト済

## LaTeXの現状（把握）
- クラス: iscs-thesis.cls を追加済み
- Goto.tex: 図は一時的にコメントアウトし、表・本文・数式は保持
- 構文修正: multirowの括弧不整合を修正済み（ビルド可能）
- Bib: `takahashi.bib` はOverleafで処理（未定義引用はBib処理で解消）

## 今後必要なもの
- データセットの確定: `training_data/{images,labels,classes.txt,data.yaml}`、分割比
- 実験スクリプト/生成物:
  - 学習済み重み（best.pt）
  - 評価CSV/PR曲線、比較表の元CSV
- 画面スクリーンショット: 各タブのUI（`Image_Goto/`に配置）
- 図差し替え: コメントアウトを戻し、`Image_Goto/`参照で統一
- 参考文献: YOLO-World含め関連研究のBibを整備、BibTeX実行

## 既知の制約/課題（把握）
- 現状、train/valを同一ディレクトリ指定（厳密評価では分割の明確化が必要）
- CPUデフォルト学習（GPU環境の設定調整で高速化）
- セグメンテーション/インスタンス分割は未実装（将来拡張）
- 推定手法の仮定（形状近似）による限界