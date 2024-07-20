
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

class GROQ(model_name="llama3-8b-8192"):

    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompt):
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
        )

        return chat_completion.choices[0].message.content