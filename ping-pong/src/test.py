import pygame
import sys
import random
import numpy as np
from random import uniform
import time
import csv
import os
import threading
import matplotlib.pyplot as plt



class RedeNeural:
    def __init__(self):
        # Inicializa pesos da rede como atributos do objeto
        self.pesosPrimeiroNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(6)])
        self.pesosSegundoNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(6)])

        self.pesosPrimeiroNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])
        self.pesosSegundoNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])

        self.pesosNeuronioDeSaida = np.array([uniform(-1, 1) for _ in range(2)])

        self.resultado = 0

    # Antes de alimentar a rede, normalize os valores
    def feedforward(self, YRaquete, XBolinha, YBola, VelocidadeX, VelocidadeY, bias=-1):
        entradas = np.array([YRaquete, XBolinha, YBola, VelocidadeX, VelocidadeY, bias])
    # Resto da função...


        self.saidaPrimeiroNeuronioCamadaEntrada = round(
            np.tanh(np.sum(entradas * self.pesosPrimeiroNeuronioCamadaEntrada)), 6)

        self.saidaSegundoNeuronioCamadaEntrada = round(
            np.tanh(np.sum(entradas * self.pesosSegundoNeuronioCamadaEntrada)), 6)

        self.saidaPrimeiroNeuronioCamadaOculta = round(
            np.tanh(np.sum(np.array([self.saidaPrimeiroNeuronioCamadaEntrada, self.saidaSegundoNeuronioCamadaEntrada]) * self.pesosPrimeiroNeuronioCamadaOculta)), 6)

        self.saidaSegundoNeuronioCamadaOculta = round(
            np.tanh(np.sum(np.array([self.saidaPrimeiroNeuronioCamadaEntrada, self.saidaSegundoNeuronioCamadaEntrada]) * self.pesosSegundoNeuronioCamadaOculta)), 6)

        self.resultado = round(self.sigmoid(np.sum(np.array([self.saidaPrimeiroNeuronioCamadaOculta, self.saidaSegundoNeuronioCamadaOculta]) * self.pesosNeuronioDeSaida)), 6)

        return self.resultado


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def atualizaPesos(self, erro, entradas, alpha=0.01):
    # Atualiza pesos da camada de saída (2 pesos)
        for i in range(len(self.pesosNeuronioDeSaida)):
            entrada_oculta = self.saidaPrimeiroNeuronioCamadaOculta if i == 0 else self.saidaSegundoNeuronioCamadaOculta
            self.pesosNeuronioDeSaida[i] += alpha * entrada_oculta * erro

        # Atualiza pesos da primeira camada oculta (2 pesos)
        for i in range(len(self.pesosPrimeiroNeuronioCamadaOculta)):
            entrada_entrada = self.saidaPrimeiroNeuronioCamadaEntrada if i == 0 else self.saidaSegundoNeuronioCamadaEntrada
            self.pesosPrimeiroNeuronioCamadaOculta[i] += alpha * entrada_entrada * erro

        # Atualiza pesos da segunda camada oculta (2 pesos)
        for i in range(len(self.pesosSegundoNeuronioCamadaOculta)):
            entrada_entrada = self.saidaPrimeiroNeuronioCamadaEntrada if i == 0 else self.saidaSegundoNeuronioCamadaEntrada
            self.pesosSegundoNeuronioCamadaOculta[i] += alpha * entrada_entrada * erro

        # Atualiza pesos da camada de entrada (agora 6 pesos)
        for i in range(len(self.pesosPrimeiroNeuronioCamadaEntrada)):
            self.pesosPrimeiroNeuronioCamadaEntrada[i] += alpha * entradas[i] * erro

        for i in range(len(self.pesosSegundoNeuronioCamadaEntrada)):
            self.pesosSegundoNeuronioCamadaEntrada[i] += alpha * entradas[i] * erro



class PongGame:
    def __init__(self, headless=False, id_thread=0, width=640, height=480):
        self.headless = headless
        self.id_thread = id_thread
        self.width = width
        self.height = height

        # Inicializar cores SEMPRE, pois são usadas na lógica também
        self.green = (0, 200, 200)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.frames_no_canto = 0
        self.erros = []  # Para salvar o erro durante o treinamento
        self.pesos_hist = []  # Para armazenar os pesos ao longo do tempo

        if not self.headless:
            pygame.init()
            self.display = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Ping Pong")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 30)
        else:
            self.display = None
            self.clock = None
            self.font = None

        # resto da inicialização... 

        self.rect_x = 272
        self.rect_y = 470
        self.x_cor = random.randint(15, width - 15)
        self.y_cor = random.randint(15, height - 15)
        self.x_change = random.randint(3, 7)
        self.y_change = random.randint(3, 7)

        self.floor_collision = False
        self.win = False
        self.sec = 0
        self.t = pygame.time.get_ticks() if not headless else 0

        self.melhor_tempo = 0
        self.partidas_jogadas = 0

        self.rede = RedeNeural()
        self.melhores_pesos = [
            self.rede.pesosPrimeiroNeuronioCamadaEntrada.copy(),
            self.rede.pesosSegundoNeuronioCamadaEntrada.copy(),
            self.rede.pesosPrimeiroNeuronioCamadaOculta.copy(),
            self.rede.pesosSegundoNeuronioCamadaOculta.copy(),
            self.rede.pesosNeuronioDeSaida.copy()
        ]

        self.historico = self.carregar_historico_csv()

    def plotar_graficos(self):
        # Gráfico de erro ao longo do tempo
        plt.figure(figsize=(10, 5))
        plt.plot(self.erros)
        plt.title("Evolução do Erro")
        plt.xlabel("Epoch")
        plt.ylabel("Erro")
        plt.grid(True)
        plt.savefig("grafico_erro.png")  # Salva o gráfico de erro
        plt.close()

        # Gráfico de pesos da rede neural ao longo do tempo
        pesos_media = [np.mean(np.array(pesos)) for pesos in self.pesos_hist]
        plt.figure(figsize=(10, 5))
        plt.plot(pesos_media)
        plt.title("Evolução dos Pesos")
        plt.xlabel("Epoch")
        plt.ylabel("Média dos Pesos")
        plt.grid(True)
        plt.savefig("grafico_pesos.png")  # Salva o gráfico de pesos
        plt.close()

    def carregar_historico_csv(self):
        historico = []
        if os.path.exists('historico.csv'):
            with open('historico.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if len(row) == 7:
                        entrada = list(map(float, row[:6]))
                        decisao = float(row[6])
                        historico.append({'entrada': entrada, 'decisao': decisao})
        return historico

    def salvar_historico_csv(self):
        filename = f"historico_{self.id_thread}.csv" if self.headless else "historico.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for registro in self.historico:
                linha = registro['entrada'] + [registro['decisao']]
                writer.writerow(linha)

    def move_player(self):
        YRaquete_norm = self.rect_x / self.width
        XBolinha_norm = self.x_cor / self.width
        YBola_norm = self.y_cor / self.height
        VelocidadeX_norm = self.x_change / 10
        VelocidadeY_norm = self.y_change / 10
        
        decisao = self.rede.feedforward(
            YRaquete_norm,
            XBolinha_norm,
            YBola_norm,
            VelocidadeX_norm,
            VelocidadeY_norm
        )
        movimento = (decisao - 0.5) * 10  # movimento entre -5 e 5

        entrada = [self.rect_x, self.x_cor, self.y_cor, self.x_change, self.y_change, -1]
        self.historico.append({'entrada': entrada, 'decisao': decisao})

        if self.rect_x <= 1:
            self.frames_no_canto += 1
            movimento = 5  # Valor fixo para direita
        elif self.rect_x >= self.width - 101:
            self.frames_no_canto += 1
            movimento = -5  # Valor fixo para esquerda
        else:
            self.frames_no_canto = 0

        if self.rect_x + movimento < 0:
            movimento = -self.rect_x
        elif self.rect_x + movimento > self.width - 100:
            movimento = (self.width - 100) - self.rect_x

        self.rect_x += movimento

    def atualizar_bola(self):
        self.x_cor += self.x_change
        self.y_cor += self.y_change

        self.x_cor = max(15, min(self.x_cor, self.width - 15))
        self.y_cor = max(15, min(self.y_cor, self.height - 15))

        if self.x_cor == 15 or self.x_cor == self.width - 15:
            self.x_change *= -1

        if self.y_cor == 15:
            self.y_change *= -1

        barra_rect = pygame.Rect(self.rect_x, self.rect_y, 100, 100)
        bola_rect = pygame.Rect(int(self.x_cor - 15), int(self.y_cor - 15), 30, 30)

        if bola_rect.colliderect(barra_rect):
            self.y_change *= -1

        if self.y_cor > self.height - 20:
            self.floor_collision = True
            self.y_change *= -1

    def treinar_rede(self):
        fitness = self.sec
        self.partidas_jogadas += 1

        if fitness >= self.melhor_tempo:
            self.melhor_tempo = fitness
            self.melhores_pesos = [
                self.rede.pesosPrimeiroNeuronioCamadaEntrada.copy(),
                self.rede.pesosSegundoNeuronioCamadaEntrada.copy(),
                self.rede.pesosPrimeiroNeuronioCamadaOculta.copy(),
                self.rede.pesosSegundoNeuronioCamadaOculta.copy(),
                self.rede.pesosNeuronioDeSaida.copy()
            ]
        else:
            (
                self.rede.pesosPrimeiroNeuronioCamadaEntrada,
                self.rede.pesosSegundoNeuronioCamadaEntrada,
                self.rede.pesosPrimeiroNeuronioCamadaOculta,
                self.rede.pesosSegundoNeuronioCamadaOculta,
                self.rede.pesosNeuronioDeSaida
            ) = self.melhores_pesos

        for registro in self.historico:
            entradas = registro['entrada']
            decisao = registro['decisao']
            movimento = decisao

            erro = (entradas[0] + movimento - entradas[2]) / 100  # Ajuste do erro

            self.rede.feedforward(*entradas)
            self.rede.atualizaPesos(erro, entradas)

            self.erros.append(erro)
            self.pesos_hist.append(self.rede.pesosNeuronioDeSaida.copy())

        self.salvar_historico_csv()
        self.historico.clear()
        self.sec = 0
        self.floor_collision = False

    def desenhar(self):
        self.display.fill(self.black)

        time_text = self.font.render(f"Tempo: {self.sec}s", True, self.green)
        self.display.blit(time_text, (10, 10))

        pygame.draw.circle(self.display, self.white, (int(self.x_cor), int(self.y_cor)), 15)
        pygame.draw.rect(self.display, self.white, (self.rect_x, self.rect_y, 100, 100))

    def loop_principal(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if pygame.time.get_ticks() - self.t >= 1000:
                self.sec += 1
                self.t = pygame.time.get_ticks()

            self.move_player()
            self.atualizar_bola()
            self.desenhar()

            if self.floor_collision:
                self.treinar_rede()

            if self.win:
                self.display.fill(self.black)
                win_text = self.font.render("You Win!", True, self.green)
                rect = win_text.get_rect(center=(self.width // 2, self.height // 2))
                self.display.blit(win_text, rect)

                # Plotar gráficos
                self.plotar_graficos()

            pygame.display.flip()
            self.clock.tick(25)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = PongGame()
    game.loop_principal()
