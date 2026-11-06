# CLIProxyAPI 本机方案

此方案直接运行开源 [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI)，在本机提供 OpenAI 兼容接口并使用操作者自己授权的订阅账号。服务不使用项目作者提供的中转服务器；正常情况下它直接访问上游，具体网络路径仍取决于本机网络和代理环境变量。下载软件及模型目录等可能访问第三方地址，OAuth 和模型请求路径请参阅上游项目。请自行确认账号适用条款和额度。

## Windows 快速开始

下面适用于曾用于实验的 CLIProxyAPI **8.0.57**。从上游官方 Release 自行下载适合系统的程序，核验来源；这里不打包第三方程序和登录凭证。PowerShell 在仓库根目录运行：

```powershell
$env:BVB_CLI_PROXY_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
$exe = 'C:\path\to\cli-proxy-api.exe'
$helper = 'methods/subscription_fork_qa/cli_proxy_local.ps1'
& $helper -Action init -Executable $exe
& $helper -Action login -Executable $exe
& $helper -Action serve -Executable $exe
```

`init` 只执行一次，产生私有配置和存放登录信息的目录（默认 `$env:USERPROFILE\.bvb-cliproxy`），不会覆盖现有配置。登录需要本人在浏览器授权，不能提交、共享配置、授权文件和本地访问密钥。`serve` 会占用当前终端，使用 Ctrl+C 停止。第二个终端设置同一个密钥，并将 `LLM_API_BASE` 设置为 `http://127.0.0.1:18318/v1`、`LLM_API_KEY` 设置为该本地密钥，再按主 README 的 SDK 例子调用。两种方案端口不同，可择一使用；模型名以本次登录后实际可用的为准。配置生成后口令存储在本机配置文件中，注意操作系统访问权限。

服务只绑定 127.0.0.1，辅助脚本不主动设定第三方中转；如果网络需要自行设置代理，先查看上游文档及现有网络环境。登录回调必须能访问本机对应端口。若直接执行脚本受 PowerShell 执行策略限制，可在当前进程临时调整策略后再运行。

## 已验证内容和限制

本项目此前曾用自己账号、在本机运行 CLIProxyAPI 做过小型真实文字和单图调用，响应成功。这个新辅助脚本的配置与启动还需在实际安装环境检查；不同版本配置格式和支持的模型可能不同。一次调用可能消耗订阅额度，优先用小图片和单题验证。该方案不是完整的视频预处理、重建或自动评分，图片须由实验代码自行发送。

CLIProxyAPI 普通 OpenAI 兼容请求可以独立调用，**并没有验证父上下文 fork 给多道题且图片只传一次**；之前 WebSocket 尝试出现上下文串题，不能当独立 fork 使用。缓存命中也不稳定，不能保证节省订阅额度。需要独立题目分支时，请看主 README 中 Codex app-server 方案及其当前真实图片测试尚未通过的限制。
