from langchain_community.llms import CTransformers

# Since we are using Hugginface.com to source our model, we will include "{Model Author}/{Model}"
# You can find this information at the top of the authors Huggingface page

model_name_or_path = "TheBloke/Pygmalion-2-7B-GGUF"

# We want to run a quantized version of the model, in this case, a 4 bit quantized model.
# We are using a quantized model because we are striving for speedy inference
# This means we must load the model into the VRAM of our GPU, you can run inference on CPU or split the
# model between GPU and CPU, but that is outside the scope of this project.

# Quantized models are great for local inference because we are limited by the VRAM capacity of our GPU
# For example, I am running this model on Windows 11, with an NVIDIA GeForce RTX 3060 that has 12GB of dedicated
# GPU memory

# You can find out how much VRAM that you have by opening
# task manager -> performance -> GPU -> look for "Dedicated GPU Memory" at the bottom

# the rule of thumb is:
# 16-bit precision (quantized) model ->
# Parameter Size = 6B : Min VRAM = ~17 GB,
# Parameter Size = 7B : Min VRAM = ~18 GB,
# Parameter Size = 13B : Min VRAM = ~26 GB

# 8-bit precision (quantized) model ->
# Parameter Size = 6B : Min VRAM = ~9 GB,
# Parameter Size = 10B : Min VRAM = ~18 GB,
# Parameter Size = 13B : Min VRAM = ~20 GB

# 4-bit precision (quantized) model ->
# Parameter Size = 6B : Min VRAM = ~5 GB,
# Parameter Size = 10B : Min VRAM = ~6 GB,
# Parameter Size = 13B : Min VRAM = ~10 GB

# I don't recommend maxing out your VRAM to run a larger model because it WILL cost you inference speed, especially if
# you plan on using "agents"

# For example, a 8bit 7B model, on my system, takes upwards of 30 seconds to generate a reply
# TheBloke on Huggingface has many high quality pre quantized models.

# To download a specific quantized model from a repository, we simply add the {Model File Name} of the quantized model
# like below:

model_file="pygmalion-2-7b.Q4_K_M.gguf"

# *IMPORTANT NOTE*
# LLM's classify input as "Tokens". Think of tokens as a hybrid between total vocabulary and memory capacity.
# For instance, most common Open Source LLM's, as of the writing of this in 2024, have a max Token Size of 2048 Tokens.
# OpenAI's models are the exception with ChatGPT3.5 having a capacity of 4096 Tokens, and ChatGPT4,
# which has an astounding 32,768 token limit!

# A Token is a unit measurements of text. A token is NOT a full word, in fact, it can be a character or a part of a
# word. It is best to think of a token as an average of 4 characters.

# As you can see we are HEAVILY limited by the max token size of any particular model.
# You will discover this as you interact with your chatbot, adding conversation history, asking long queries,
# or generating long responses.

# We will cover ways to overcome token limits later on.


# Now we need to fine tune our model:

# gpu_layers = The number of layers to run on GPU, after experimenting, 50 is the sweet spot on my system.
# Too many = slow inference/errors, Too little = slow inference

# context_length = How many tokens should the LLM commit to remembering information

# max_new_tokens = How many tokens should the LLM limit itself to when generating a new response

# top_k = Sample from the k most likely next tokens at each step. (Lower values make LLM concentrate on sampling of the
# highest probability tokens for each step)

# top_p = The cumulative probability cutoff for token selection. (Lower values make LLM reduce diversity and focus more
# on probable tokens)

# temperature = Controls randomness. (Higher values make LLM increase diversity of replies)

# stop = a list "['str']" of sequences to stop generation when encountered. For instance, if your chatbot is generating
# its AI reply and then hallucinating a human reply like {Human: blah blah} set stop to ["Human:"]

# stream = A boolean stating whether the LLM should stream its generated tokens, as they are generated

# reset = A boolean stating whether the LLM should reset the model state before generating text. Useful if you don't
# plan on carrying on conversations with LLM

# repetition_penalty = Penalize the LLM for generating repeat tokens. Default = 1.1

config = {
    "gpu_layers": 50,
    "context_length": 1024,
    "max_new_tokens": 1024,
    "top_k": 10,
    "top_p": 10,
    "temperature": 0,
}


# YOU MUST HAVE CUDA TOOLKIT INSTALLED! I am running 11.8 (https://developer.nvidia.com/cuda-11-8-0-download-archive)
# YOU WILL ALSO NEED TORCH WITH CUDA DEPENDENCIES! (https://pytorch.org/get-started/locally/)
# * pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 *
# To run inference on a CUDA GPU with CTransformers, ensure you downloaded the cuda version of ctransformers
# pip install ctransformers[cuda]

# model = {Model Author}/{Model} from above

# model_file = {Model File Name} from above

# model_type = (*OPTIONAL*) The original models name. Since we are running Pygmalion 2 7B, and it is a LoRa of Meta's
# Llama LLM, we will include "llama"

# cache_dir = The location that the cached model will be downloaded to. The default location is ~/.cache/huggingface/


llm = CTransformers(
    model=model_name_or_path,
    model_file=model_file,
    model_type="llama",
    cache_dir="../workspace/",
    config=config,
)