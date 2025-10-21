# Self-Hosted Runner クイックスタートガイド

## 現在の問題
GitHub Actionsで「Waiting for a runner to pick up this job...」と表示され、ジョブが実行されない状態です。

## 即座に解決する方法

### 方法1: GitHub-hosted runnerを使用（推奨）
1. `.github/workflows/commit-hash-display-github-hosted.yml` を有効化
2. 元の `commit-hash-display.yml` を一時的に無効化
3. これで即座にワークフローが実行されます

### 方法2: Self-hosted runnerを設定

#### ステップ1: GitHubリポジトリでrunnerを追加
1. GitHubリポジトリの **Settings** → **Actions** → **Runners** に移動
2. **New runner** をクリック
3. **Self-hosted** を選択
4. **Windows** を選択
5. 表示されるコマンドをコピー

#### ステップ2: Windowsマシンでrunnerを設定
```powershell
# PowerShellを管理者権限で開く
# 1. ダウンロード（GitHubが提供する実際のURLを使用）
Invoke-WebRequest -Uri "https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-win-x64-2.311.0.zip" -OutFile "actions-runner-win-x64-2.311.0.zip"

# 2. 解凍
Expand-Archive -Path "actions-runner-win-x64-2.311.0.zip" -DestinationPath "actions-runner"

# 3. 設定（GitHubが提供する実際のコマンドを使用）
cd actions-runner
./config.cmd --url https://github.com/YOUR_USERNAME/YOUR_REPOSITORY --token YOUR_TOKEN

# 4. 起動
./run.cmd
```

#### ステップ3: 確認
- GitHubリポジトリの **Settings** → **Actions** → **Runners** で
- runnerが **Online** 状態になっていることを確認

## トラブルシューティング

### Runnerが表示されない場合
- GitHubのトークンが正しいか確認
- ネットワーク接続を確認
- ファイアウォール設定を確認

### RunnerがOfflineの場合
```powershell
cd actions-runner
./run.cmd
```

### サービスとして常時起動したい場合
```powershell
cd actions-runner
./svc.cmd install
./svc.cmd start
```

## 推奨事項
- **開発・テスト用途**: GitHub-hosted runnerを使用
- **本番・機密データ**: self-hosted runnerを使用
- **コスト考慮**: GitHub-hosted runnerは無料枠内で使用可能
