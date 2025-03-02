import base64
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from huggingface_hub import InferenceClient


def convert_img_text(url):
    image_to_text = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    output = image_to_text(url)
    return output[0]["generated_text"]


def generate_story(scenario):
    TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(token=TOKEN)

    completion = client.text_generation(
        prompt=f"Write a short story based on this scenario: {scenario}",
        model="tiiuae/falcon-7b-instruct",  # Modern and powerful instruction-tuned model
        max_new_tokens=200,
        temperature=0.7,
    )

    return completion


def text_to_speech(text):
    TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {"inputs": text}

    response = requests.post(API_URL, headers=headers, json=payload)
    with open("output.wav", "wb") as f:
        f.write(response.content)


def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


def translate_to_indic_per_line(story, language="ml"):
    english_to_hindi = pipeline(
        "translation", model=f"Helsinki-NLP/opus-mt-en-{language}"
    )
    output = english_to_hindi(story)
    return output


def translate_to_indic(story, language="ml"):
    lines = story.split(".")[:-1]
    translated_story = []
    for line in lines:
        output = translate_to_indic_per_line(line, language)
        translated_story.append(output[0]["translation_text"])
    story_in_indic = "".join(translated_story)
    return story_in_indic


def generate_text_to_image(scenario):
    TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.content
        st.write(f"Status code: {response.status_code}")
        st.write(f"Response: {response.text}")
        return None

    image_bytes = query(
        {
            "inputs": scenario,
        }
    )

    return image_bytes


def main():
    scenario = ""
    st.set_page_config(page_title="Using LLM", page_icon="🤖")
    st.header(
        "A Langchain example to convert image to text and generate a story and read it out loud"
    )
    upload_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if upload_image is not None:
        print(upload_image)

        bytes_data = upload_image.getvalue()
        with open(upload_image.name, "wb") as f:
            f.write(bytes_data)

        st.image(upload_image, caption="Uploaded Image.", use_column_width=True)

        with st.expander("Generated text"):
            scenario = convert_img_text(upload_image.name)
            st.write(scenario)

        with st.expander("Generated story"):
            story = generate_story(scenario)
            st.write(story)

        with st.expander("Translate to Hindi"):
            story_in_indic = translate_to_indic(story, language="hi")
            st.write(story_in_indic)

        with st.expander("Translate to Malayalam"):
            story_in_indic = translate_to_indic(story, language="ml")
            st.write(story_in_indic)

        st.write("Generating audio...")

        with st.expander("Generated audio"):
            text_to_speech(story)

        st.write("Done!")

    st.write("# Play Audio!")
    autoplay_audio("output.wav")

    with st.expander("Generate Image"):
        image_bytes = generate_text_to_image(scenario)
        if image_bytes:
            st.image(image_bytes, caption="Generated Image.")
        else:
            st.write("Image generation failed. Please try again.")


if __name__ == "__main__":
    load_dotenv()
    main()
