from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("captcha.png")

client = OpenAI(
    base_url="http://10.21.16.109:11434/v1",
    api_key="ollama",
)

completion = client.chat.completions.create(
    extra_body={},
    model="qwen3.5:35b",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Identify the alphanumeric characters in this captcha image. Return only the characters without any spaces, punctuation, or explanation."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)
print(completion)
print(completion.choices[0].message.content.strip())