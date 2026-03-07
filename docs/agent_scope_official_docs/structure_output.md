### 结构化输出

AgentScope 中的结构化输出是与工具调用紧密结合的。具体来说，`ReActAgent` 类在其 `__call__` 函数中接收 `pydantic.BaseModel` 的子类作为 `structured_model` 参数。 从而提供复杂的结构化输出限制。 然后我们可以从 返回消息的 `metadata` 字段获取结构化输出。

以介绍爱因斯坦为例：
```python
# 创建一个 ReAct 智能体
agent = ReActAgent(
    name="Jarvis",
    sys_prompt="你是一个名为 Jarvis 的有用助手。",
    model=DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)


# 结构化模型
class Model(BaseModel):
    name: str = Field(description="人物的姓名")
    description: str = Field(description="人物的一句话描述")
    age: int = Field(description="年龄")
    honor: list[str] = Field(description="人物荣誉列表")


async def example_structured_output() -> None:
    """结构化输出示例"""
    
    #这个说白了就是agent(Msg信息，结构化模型)就是把msg传给agent，强制要求用指定model输出 
    res = await agent(
        Msg(
            "user",
            "介绍爱因斯坦",
            "user",
        ),
        structured_model=Model,
    )
    print("\n结构化输出：")
    print(json.dumps(res.metadata, indent=4, ensure_ascii=False))


asyncio.run(example_structured_output())
```
