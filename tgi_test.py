from text_generation import InferenceAPIClient
from text_generation.inference_api import deployed_models

print(deployed_models())
client = InferenceAPIClient("bigscience/bloomz")
text = client.generate("Why is the sky blue?").generated_text
print(text)
# ' Rayleigh scattering'

# Token Streaming
text = ""
for response in client.generate_stream("Why is the sky blue?"):
    if not response.token.special:
        text += response.token.text

print(text)