import pygame
import sys
import random
import numpy as np
from random import uniform
import time
import csv
import os
import threading



class RedeNeural:
    def __init__(self):
        # Inicializa pesos da rede como atributos do objeto
        self.pesosPrimeiroNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(6)])
        self.pesosSegundoNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(6)])

        self.pesosPrimeiroNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])
        self.pesosSegundoNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])

        self.pesosNeuronioDeSaida = np.array([uniform(-1, 1) for _ in range(2)])

        self.resultado = 0

    def feedforward(self, YRaquete, XBolinha, YBola, VelocidadeX, VelocidadeY, bias=-1):
        entradas = np.array([YRaquete, XBolinha, YBola, VelocidadeX, VelocidadeY, bias])

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

        # resto da inicialização (variáveis, rede neural, histórico)
        # exemplo:
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

    def carregar_historico_csv(self):
        historico = []
        if os.path.exists('historico.csv'):
            with open('historico.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # Cada linha: [rect_x, x_cor, y_cor, x_change, y_change, bias, decisao]
                    if len(row) == 7:
                        entrada = list(map(float, row[:6]))
                        decisao = float(row[6])
                        historico.append({'entrada': entrada, 'decisao': decisao})
        return historico

        return historico
    def salvar_historico_csv(self):
        filename = f"historico_{self.id_thread}.csv" if self.headless else "historico.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for registro in self.historico:
                linha = registro['entrada'] + [registro['decisao']]
                writer.writerow(linha)


    def move_player(self):
        decisao = self.rede.feedforwarddecisao = self.rede.feedforward(
            self.rect_x,      # YRaquete
            self.x_cor,       # XBolinha
            self.y_cor,       # YBola
            self.x_change,    # VelocidadeX
            self.y_change     # VelocidadeY
        )

        movimento = (decisao - 0.5) * 10  # movimento entre -5 e 5

        entrada = [self.rect_x, self.x_cor, self.y_cor, self.x_change, self.y_change, -1]
        self.historico.append({'entrada': entrada, 'decisao': decisao})

        # Controle de frames no canto e força movimento depois de um tempo parado
        if self.rect_x <= 5:
            self.frames_no_canto += 1
            if self.frames_no_canto > 10:
                movimento = 2  # força mover para direita
        elif self.rect_x >= self.width - 105:
            self.frames_no_canto += 1
            if self.frames_no_canto > 10:
                movimento = -2  # força mover para esquerda
        else:
            self.frames_no_canto = 0

        # Limitar para não sair da tela
        if self.rect_x <= 0 and movimento < 0:
            movimento = 0
        elif self.rect_x >= self.width - 100 and movimento > 0:
            movimento = 0

        self.rect_x += movimento
        self.rect_x = max(0, min(self.rect_x, self.width - 100))

        print(f"Posição raquete: {self.rect_x}, Frames no canto: {self.frames_no_canto}, Movimento: {movimento}")





    def atualizar_bola(self):
        self.x_cor += self.x_change
        self.y_cor += self.y_change

        # Limita a bola dentro dos limites da janela, para evitar coordenadas inválidas
        self.x_cor = max(15, min(self.x_cor, self.width - 15))
        self.y_cor = max(15, min(self.y_cor, self.height - 15))

        # Colisão com paredes laterais
        if self.x_cor == 15 or self.x_cor == self.width - 15:
            self.x_change *= -1

        # Colisão com topo
        if self.y_cor == 15:
            self.y_change *= -1

        # Retângulo da barra (raquete)
        barra_rect = pygame.Rect(self.rect_x, self.rect_y, 100, 100)

        # Retângulo da bola: centro na posição, tamanho 30x30
        bola_rect = pygame.Rect(int(self.x_cor - 15), int(self.y_cor - 15), 30, 30)

        # Colisão da bola com a barra
        if bola_rect.colliderect(barra_rect):
            self.y_change *= -1

        # Bola passou da barra (chão)
        if self.y_cor > self.height - 20:
            self.floor_collision = True
            self.y_change *= -1


    def treinar_rede(self):
        fitness = self.sec
        self.partidas_jogadas += 1

        if fitness >= self.melhor_tempo:
            self.melhor_tempo = fitness
            # Salva pesos atuais
            self.melhores_pesos = [
                self.rede.pesosPrimeiroNeuronioCamadaEntrada.copy(),
                self.rede.pesosSegundoNeuronioCamadaEntrada.copy(),
                self.rede.pesosPrimeiroNeuronioCamadaOculta.copy(),
                self.rede.pesosSegundoNeuronioCamadaOculta.copy(),
                self.rede.pesosNeuronioDeSaida.copy()
            ]
        else:
            # Carrega melhores pesos
            (
                self.rede.pesosPrimeiroNeuronioCamadaEntrada,
                self.rede.pesosSegundoNeuronioCamadaEntrada,
                self.rede.pesosPrimeiroNeuronioCamadaOculta,
                self.rede.pesosSegundoNeuronioCamadaOculta,
                self.rede.pesosNeuronioDeSaida
            ) = self.melhores_pesos

        # Treina com todo o histórico acumulado
        for registro in self.historico:
            entradas = registro['entrada']
            decisao = registro['decisao']
            movimento = decisao

            penalidade_canto = 0
            if self.frames_no_canto > 30:
                penalidade_canto = 1.0  # penalidade maior para forçar sair do canto

            erro = (entradas[0] + movimento - entradas[2]) / 100 + penalidade_canto

            self.rede.feedforward(*entradas)
            self.rede.atualizaPesos(erro, entradas)

        


            # Chama feedforward com todas as 6 entradas
            self.rede.feedforward(*entradas)

            # Atualiza os pesos passando o erro e as entradas completas
            self.rede.atualizaPesos(erro, entradas)

        self.salvar_historico_csv()
        self.historico.clear()
        self.sec = 0
        self.floor_collision = False


    def desenhar(self):
        self.display.fill(self.black)

        # Texto tempo
        time_text = self.font.render(f"Tempo: {self.sec}s", True, self.green)
        self.display.blit(time_text, (10, 10))

        # Bola
        pygame.draw.circle(self.display, self.white, (int(self.x_cor), int(self.y_cor)), 15)

        # Barra jogador
        pygame.draw.rect(self.display, self.white, (self.rect_x, self.rect_y, 100, 100))

    def loop_principal(self):
        running = True
        while running:
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Atualiza tempo
            if pygame.time.get_ticks() - self.t >= 1000:
                self.sec += 1
                self.t = pygame.time.get_ticks()
                if self.sec >= 60:
                    self.win = True
                if self.floor_collision:
                    self.sec -= 1  # Não conta se perdeu

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

            pygame.display.flip()
            self.clock.tick(25)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = PongGame()
    game.loop_principal()
    

