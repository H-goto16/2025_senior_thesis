# 🍽️ Dish Detection - Advanced Model Management System

高機能なYOLO物体検出モデル管理アプリケーション。トレーニング、可視化、ラベリング機能を統合したオールインワンソリューション。

## ✨ 主な機能

### 🎯 物体検出
- リアルタイム物体検出
- カスタム信頼度閾値設定
- バウンディングボックス可視化
- 複数画像形式対応

### 🏷️ ラベリング機能
- 直感的なドラッグ&ドロップラベリング
- マルチクラス対応
- YOLO形式での自動保存
- ラベル編集・削除機能

### 🚀 モデルトレーニング
- バックグラウンドトレーニング
- リアルタイム進捗監視
- 自動モデル保存・読み込み
- トレーニング履歴管理

### 📊 高度な可視化
- インタラクティブなトレーニングメトリクス
- モデル性能比較
- データセット分析
- Plotlyベースの動的グラフ

### 🎨 ユーザーエクスペリエンス
- ダークテーマ対応
- レスポンシブデザイン
- Android/iOS/Web対応
- 直感的なナビゲーション

## 🏗️ アーキテクチャ

### バックエンド (FastAPI)
```
backend/
├── src/
│   ├── main.py                 # FastAPIアプリケーション
│   ├── yolo/
│   │   └── object_detection.py # YOLO検出エンジン
│   └── training/
│       └── training_manager.py # トレーニング管理
├── requirements.txt            # Python依存関係
└── venv/                      # 仮想環境
```

### フロントエンド (React Native + Expo)
```
frontend/
├── src/
│   ├── app/                   # Expo Router画面
│   │   ├── (tabs)/
│   │   │   ├── index.tsx      # 物体検出画面
│   │   │   ├── labeling.tsx   # ラベリング画面
│   │   │   ├── training.tsx   # トレーニング管理
│   │   │   ├── models.tsx     # モデル管理
│   │   │   └── analytics.tsx  # 分析・可視化
│   │   └── _layout.tsx        # ルートレイアウト
│   ├── lib/
│   │   ├── api.ts            # APIクライアント
│   │   ├── store.ts          # Zustand状態管理
│   │   └── utils.ts          # ユーティリティ
│   └── components/           # 再利用可能コンポーネント
├── package.json              # Node.js依存関係
└── app.config.js            # Expo設定
```

## 🚀 セットアップ

### 前提条件
- Python 3.12+
- Node.js 18+
- pnpm

### バックエンドセットアップ
```bash
cd backend

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt

# サーバー起動
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### フロントエンドセットアップ
```bash
cd frontend

# 依存関係インストール
pnpm install

# 開発サーバー起動
pnpm start

# プラットフォーム別起動
pnpm android  # Android
pnpm ios      # iOS
pnpm web      # Web
```

## 📱 使用方法

### 1. 物体検出
1. **Detection**タブを開く
2. カメラまたはギャラリーから画像を選択
3. 検出クラスを設定
4. 検出実行

### 2. ラベリング
1. **Labeling**タブを開く
2. 画像を選択
3. ラベルを選択
4. ドラッグしてバウンディングボックスを描画
5. ラベルを送信

### 3. モデルトレーニング
1. **Training**タブを開く
2. エポック数を設定
3. トレーニング開始
4. 進捗をリアルタイム監視

### 4. 分析・可視化
1. **Analytics**タブを開く
2. データセット統計を確認
3. モデル性能を比較
4. トレーニング履歴を分析

## 🔧 API エンドポイント

### 物体検出
- `POST /detect` - 画像から物体検出
- `POST /detect/with-confidence` - 信頼度指定検出

### モデル管理
- `GET /model/info` - モデル情報取得
- `GET /model/classes` - 検出クラス一覧
- `POST /model/classes` - 検出クラス追加
- `GET /models/available` - 利用可能モデル一覧
- `POST /models/load/{path}` - モデル読み込み
- `POST /models/backup` - モデルバックアップ

### トレーニング
- `POST /training/start-async` - 非同期トレーニング開始
- `GET /training/status` - トレーニング状況取得
- `GET /training/history` - トレーニング履歴
- `GET /training/metrics/{run_name}` - 詳細メトリクス

### データ管理
- `POST /labeling/submit` - ラベリングデータ送信
- `GET /data/analysis` - データセット分析
- `GET /data/export` - データエクスポート
- `DELETE /training/cleanup` - 古いトレーニング削除

## 🎨 技術スタック

### バックエンド
- **FastAPI** - 高性能WebAPIフレームワーク
- **Ultralytics YOLO** - 最新の物体検出モデル
- **Plotly** - インタラクティブ可視化
- **Pandas/NumPy** - データ処理
- **Matplotlib/Seaborn** - 統計可視化

### フロントエンド
- **React Native** - クロスプラットフォーム開発
- **Expo** - 開発・デプロイメントプラットフォーム
- **Zustand** - 軽量状態管理
- **TanStack Query** - サーバー状態管理
- **NativeWind** - Tailwind CSS for React Native
- **React Native Reanimated** - 高性能アニメーション

## 🔄 開発ワークフロー

1. **データ収集** - カメラ/ギャラリーから画像取得
2. **ラベリング** - 直感的なUI でアノテーション作成
3. **トレーニング** - バックグラウンドでモデル学習
4. **評価** - 詳細メトリクスで性能分析
5. **デプロイ** - 最適なモデルを本番環境へ

## 🚨 トラブルシューティング

### よくある問題

**ポートエラー**
```bash
# 別のポートを使用
uvicorn src.main:app --port 8002
```

**モジュールインポートエラー**
```bash
# 仮想環境の確認
source venv/bin/activate
pip list
```

**権限エラー**
```bash
# カメラ・ストレージ権限を確認
# アプリ設定で権限を有効化
```

## 📈 パフォーマンス最適化

- **GPU加速**: CUDA対応環境でGPUトレーニング
- **バッチ処理**: 複数画像の並列処理
- **キャッシュ**: React Queryによる効率的データ取得
- **レスポンシブ**: 画面サイズに応じた最適化

## 🤝 コントリビューション

1. フォークを作成
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチをプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 物体検出モデル
- [FastAPI](https://fastapi.tiangolo.com/) - WebAPIフレームワーク
- [Expo](https://expo.dev/) - React Native開発プラットフォーム
- [Plotly](https://plotly.com/) - インタラクティブ可視化

---

**🚀 研究・開発用の高機能モデル管理システムで、効率的な機械学習ワークフローを実現！**