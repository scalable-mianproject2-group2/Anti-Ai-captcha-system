import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyA7av0Tks3t54kip2AvosF1mQGsr8UpchM")

for m in genai.list_models():
    print(m.name, " | supports: ", m.supported_generation_methods)
