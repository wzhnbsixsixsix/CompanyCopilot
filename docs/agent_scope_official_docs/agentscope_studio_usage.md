
### 快速打开Studio

AgentScope Studio 通过 `npm` 安装：

```shell
npm install -g @agentscope/studio
```

使用以下命令启动 Studio：

```sh
as_studio
```



要将应用程序连接到 Studio，请在 `agentscope.init` 函数中使用 


`studio_url` 参数：

```py
import agentscope

agentscope.init(studio_url="http://localhost:3000")
#agentscope.init(project,studio_url="http://localhost:3000")
#加入project参数可以在studio里面看到是哪个项目在跑

# 应用程序代码
...
```

