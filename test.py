from openai import OpenAI
import json
client = OpenAI(
 api_key="dummy",
 base_url="http://134.100.39.10:30001/v1",
)
resp = client.chat.completions.create(
 model="openai/gpt-oss-120b",
 messages=[
     {"role": "system", "content": "You are a helpful assistant."},
     {"role": "user",   "content": "Translate 'Good morning' into German."},
 ],
)

print(resp.choices[0].message.content)