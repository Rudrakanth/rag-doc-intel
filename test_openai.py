from src.config import openai_client, AZURE_OPENAI_CHAT_DEPLOYMENT
print(openai_client.base_url)
r = openai_client.chat.completions.create(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT,
    messages=[{"role": "user", "content": "hello"}]
)

print(r.choices[0].message.content)
