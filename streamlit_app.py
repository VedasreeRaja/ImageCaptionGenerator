import pickle
from model import BahdanauAttention, EncoderCNN, Decoder
from vocab import Vocab_Builder
import torch
import nltk
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
import time
import streamlit as st
import requests
import os
from io import BytesIO
from nltk.util import ngrams
from nltk.metrics import *
from nltk.translate.bleu_score import sentence_bleu
import wget
reference = []
with open('captions.txt', 'r') as file:
    for line in file:
        reference.append(line.strip().split()) 
device = 'cpu'
st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Image Caption Generator"
)
def transform_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    )
    return transform(image)
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
@st.cache
def download_data():
    path1 = './Flickr30k_Decoder_10.pth.tar'
    path2 = './resnet5010.pt'
    if not os.path.exists(path1):
        decoder_url = 'wget -O ./Flickr30k_Decoder_10.pth.tar https://www.dropbox.com/s/cf2ox65vi7c2fou/Flickr30k_Decoder_10.pth.tar?dl=0'
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(decoder_url)
    else:
        print("Model 1 is here.")
    if not os.path.exists(path2):
        encoder_url = 'wget -O ./resnet5010.pt https://www.dropbox.com/s/v0ikcdbh8w2rqii/resnet5010.pt?dl=0'
        with st.spinner('Downloading model weights for resnet50'):
            os.system(encoder_url)
    else:
        print("Model 2 is here.")
@st.cache
def load_model(): 
    vocab = Vocab_Builder(freq_threshold = 5)
    vocab_path = './vocab.pickle'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(len(vocab))
    embed_size = 350
    encoder_dim = 1024
    decoder_dim = 512
    attention_dim = 512
    vocab_size = len(vocab)
    learning_rate = 4e-5 
    resnet_path = './resnet5010.pt'
    encoder = EncoderCNN()
    encoder.load_state_dict( torch.load( resnet_path, map_location = 'cpu') )
    encoder.to(device)
    encoder.eval() 
    decoder_path = './Flickr30k_Decoder_10.pth.tar'
    decoder = Decoder(encoder_dim, decoder_dim, embed_size, vocab_size, attention_dim, device)    
    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    checkpoint = torch.load(decoder_path,map_location='cpu')
    decoder.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    decoder = decoder.to(device)
    decoder.eval()
    return vocab, encoder, decoder
def predict_caption(image_bytes):
    captions = []
    img_t = transform_image(image_bytes)
    for i in range(1,6):
        encoded_output = encoder(img_t.unsqueeze(0).to(device))
        caps = decoder.beam_search(encoded_output,i)
        caps = caps[1:-1]
        caption = [vocab.itos[idx] for idx in caps]
        caption = ' '.join(caption)
        print(caption)
        captions.append(caption)
#    for i in range(len(captions)):
 #       s = ("** Caption " + str(i + 1) + ": " + captions[i] + "**")
#      st.markdown(s)        
    return captions
@st.cache(ttl=3600, max_entries=10)
def load_output_image(img):
    if isinstance(img, str): 
        image = Image.open(img)
    else:
        img_bytes = img.read() 
        image = Image.open(io.BytesIO(img_bytes) ).convert("RGB") 
    image = ImageOps.exif_transpose(image) 
    return image
@st.cache(ttl=3600, max_entries=10)
def pypng():
    image = Image.open('data/logo.png')
    return image 
def longest_common_subsequence(tokens1, tokens2):
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i - 1] == tokens2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
def rouge_l_f1(hypothesis, reference):
    hypothesis_tokens = hypothesis.split()
    reference_tokens = reference.split()
    lcs_length = longest_common_subsequence(hypothesis_tokens, reference_tokens)
    precision = lcs_length / len(hypothesis_tokens)
    recall = lcs_length / len(reference_tokens)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    return f1_score
with open('captions.txt', 'r') as file:
    reference_texts = [line.strip() for line in file]
if __name__ == '__main__':
    download_data()
    vocab, encoder, decoder = load_model()
    logo_image = pypng()
    st.image(logo_image, width = 500)
    st.title("Image Caption Generator")
    st.text("") 
    st.success("Welcome to Image Caption Generator! Please upload an image!")   
    args = { 'sunset' : 'imgs/sunset.jpeg' }
    img_upload  = st.file_uploader(label= 'Upload Image', type = ['png', 'jpg', 'jpeg','webp'])
    img_open = args['sunset'] if img_upload is None else img_upload
    image = load_output_image(img_open)
    st.image(image,use_column_width=True)
    if st.button('Generate captions!'):
        sentences = predict_caption(image)
        sum = 0
        sum2 = 0
        bleu = []
        for predicted_caption in sentences:
            predicted_caption_tokens = predicted_caption.split()
            s1 = ("** Generated Caption : " + predicted_caption + "**")
            st.markdown(s1)   
            bleu_score = sentence_bleu(reference, predicted_caption_tokens)
            sum+=bleu_score
        average = sum/5
        s2 = ('**Average BLEU score -> {:.4f}**'.format(average))
        st.markdown(s2)
        hypothesis_text = predicted_caption
        rouge_l_scores = []
        for reference_text in reference_texts:
            rouge_l_score = rouge_l_f1(hypothesis_text, reference_text)
            rouge_l_scores.append(rouge_l_score)
            sum2 += rouge_l_score
        average_rouge_l = sum2 / len(rouge_l_scores)
        max_rouge_l = max(rouge_l_scores)
        s3 =  (f"**Maximum ROUGE-L Score (F1) for {len(reference_texts)} reference sentences: {max_rouge_l:.4f}**")
        st.markdown(s3)
        s3 =  (f"**Average ROUGE-L Score (F1) for {len(reference_texts)} reference sentences: {average_rouge_l:.4f}**")
        st.markdown(s3)
        for predicted_caption in sentences:
            predicted_caption_tokens = predicted_caption.split()
            bleu_score = sentence_bleu(reference, predicted_caption_tokens)
            bleu.append(bleu_score)
        plt.figure(figsize=(5, 7))
        plt.bar(range(len(bleu)), bleu, tick_label=[f"Caption {i+1}" for i in range(len(bleu))],color='red')
        plt.xlabel("Generated Captions")
        plt.ylabel("BLEU Score")
        plt.title("BLEU Scores for Generated Captions")
        st.pyplot(plt)
        st.success("Click again to retry or try a different image by uploading")
        st.balloons()
