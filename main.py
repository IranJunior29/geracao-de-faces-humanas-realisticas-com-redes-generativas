# Imports
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

''' Construindo o Generator '''

# Classe do generator
class Generator(nn.Module):

    # Método Construtor
    def __init__(self):

        super(Generator, self).__init__()

        self.main = nn.Sequential(

            # Input
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # Size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # Size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # Método Forward
    def forward(self, input):
        return  self.main(input)

''' Construindo o Discriminator '''

# Classe do Discriminator
class Discriminator(nn.Module):

    # Método Construtor
    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            # Input é (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # Método Forward
    def forward(self, input):
        return self.main(input)

if __name__ == '__main__':

    ''' Verificando o Ambiente de Desenvolvimento '''

    # Verifica se uma GPU está disponível e define o dispositivo apropriado
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define o device (GPU ou CPU)
    device = torch.device(processing_device)
    print(device)

    ''' Definindo Parâmetros Globais '''

    # Seed para reproduzir o mesmo resultado
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Diretório raiz dos dados (Dataset foi retirado do site http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
    dataroot = "dados/celeba"

    # Número de workes para o dataloader
    workers = 2

    ''' Preparando a Pasta de Imagens e o Dataloader '''

    # Todas as imagens serão redimensionadas para este tamanho usando um transformador
    image_size = 64

    # Cria o dataset com a pasta de imagens
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))

    # Batch size para o treinamento
    batch_size = 128

    # Cria o dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=workers)

    # Inicialização customizada dos pesos nas redes dsanetG e dsanetD
    def init_pesos(m):

        classname = m.__class__.__name__

        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    # Número de canais de cores nas imagens de treinamento
    nc = 3

    # Tamanho de vetor latente z (ou seja, tamanho da entrada do gerador)
    nz = 100

    # Tamanho dos mapas de recursos no gerador
    ngf = 64

    # Tamanho dos mapas de recursos no discriminador
    ndf = 64

    # Cria o gerador
    netG = Generator().to(device)

    # Inicializa os pesos
    netG.apply(init_pesos)

    # Cria o discriminador
    netD = Discriminator().to(device)

    # Inicializa os pesos
    netD.apply(init_pesos)

    ''' Função de Perda e Otimizador '''

    #Inicializa a função de erro BCELoss
    criterion = nn.BCELoss()

    # Taxa de aprendizado
    lr = 0.0002

    # Hiperparâmetros do otimizador Adam
    beta1 = 0.5
    beta2 = 0.999

    # Otimizador Adam para ambas as redes G e D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    ''' Loop de Treinamento e Avaliação '''

    # Número de passadas de treino
    num_epochs = 5

    # Estabelece conveções para rótulos reais e falsos durante o treinamento
    real_label = 1.
    fake_label = 0.

    # Cria um lote de vetores latentes que usaremos para visualizar a progressão do gerador (para avaliação do modelo)
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Listas para acompanhar o progresso
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print('Iniciando o Treinamento...')

    # Loop por cada época
    for epoch in range(num_epochs):

        # Loop por cada batch do dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Atualiza a rede D maximizando: log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Zera os gradientes
            netD.zero_grad()

            # Prepara o batch de dados
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass de dados reais pela rede D
            output = netD(real_cpu).view(-1)

            # Calcula o erro
            errD_real = criterion(output, label)

            # Calcula os gradientes da rede D no backward pass para os dados reais
            errD_real.backward()
            D_x = output.mean().item()

            # Gera o vetor latente
            noise = torch.randn(b_size, nz, 1, 1, device=device)

            # Gera imagens fake com a rede G
            fake = netG(noise)
            label.fill_(fake_label)

            # Classifica as imagens fake com a rede D
            output = netD(fake.detach()).view(-1)

            # Calcula o erro de D com as imagens fake
            errD_fake = criterion(output, label)

            # Calcula os gradientes para cada batch nos dados fake
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # O erro da rede D é a soma dos erros com dados reais e com dados fake
            errD = errD_real + errD_fake

            # Atualiza os pesos da rede D
            optimizerD.step()

            ############################
            # (2) Atualiza a rede G maximizando: log(D(G(z)))
            ###########################

            # Zera os gradientes
            netG.zero_grad()

            # Labels fake são preenchidos com labels reais
            label.fill_(real_label)

            # Como acabamos de atualizar D, executamos outra passagem de lote de imagens fake por D
            output = netD(fake).view(-1)

            # Calculamos a perda de G com base nesta saída anterior
            errG = criterion(output, label)

            # Calculamos os gradientes de G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Atualiza os pesos em G
            optimizerG.step()

            # Estatística de treino
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item()))

            # Salva os erros para criar um plot mais tarde
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Verificamos como o gerador está salvando a saída de G em fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        print("Treinamento Concluído!")




























