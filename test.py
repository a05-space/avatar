from openai import OpenAI

client = OpenAI(
    base_url="http://47.98.41.23:11434/v1",
    api_key="ollama",
)

completion = client.chat.completions.create(
    model="qwen3.5:35b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "你认识猪小屁嘛？"
                }
            ]
        }
    ]
)
print(completion)
print()
print(completion.choices[0].message.content.strip())