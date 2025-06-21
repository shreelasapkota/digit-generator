import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# VAE architecture (must match training script)
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28*28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

@st.cache(allow_output_mutation=True)
def load_model():
    model = VAE()
    model.load_state_dict(torch.load("vae_mnist.pth", map_location='cpu'))
    model.eval()
    return model

def generate_images(model, num_images=5):
    with torch.no_grad():
        z = torch.randn(num_images, 20)
        samples = model.decode(z).view(-1, 28, 28)
    return samples

st.title("Handwritten Digit Generator")

digit = st.slider("Select digit (0-9)", 0, 9, 0)

model = load_model()

if st.button("Generate 5 Images"):
    images = generate_images(model, 5)
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
