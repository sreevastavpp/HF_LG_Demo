import base64
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI


def convert_img_text(url):
    image_to_text = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
    )
    output = image_to_text(url)
    return output[0]["generated_text"]


def generate_story(scenario):
    template = """You are a story teller:
    You can generate a short story based on a simple narritave, the story should be no more than 40 words:
    
    CONTEXT:{scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    llm = OpenAI()

    story_llm = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
    )

    story = story_llm.run(scenario=scenario)
    return story


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


def main():
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

        st.write("Generating audio...")

        with st.expander("Generated audio"):
            text_to_speech(story)

        st.write("Done!")

    st.write("# Play Audio!")
    autoplay_audio("output.wav")


if __name__ == "__main__":
    load_dotenv()
    main()
