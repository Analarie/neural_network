import pygame
import sys
import random
import numpy as np
from random import uniform
import time

class RedeNeural:
    def __init__(self):
        # Inicializa pesos da rede como atributos do objeto
        self.pesosPrimeiroNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(4)])
        self.pesosSegundoNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(4)])

        self.pesosPrimeiroNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])
        self.pesosSegundoNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])

        self.pesosNeuronioDeSaida = np.array([uniform(-1, 1) for _ in range(2)])

        self.resultado = 0

    def feedforward(self, YRaquete, XBolinha, YBola, bias=-1):
        entradas = np.array([YRaquete, XBolinha, YBola, bias])

        self.saidaPrimeiroNeuronioCamadaEntrada = round(
            np.tanh(np.sum(entradas * self.pesosPrimeiroNeuronioCamadaEntrada)), 6)

        self.saidaSegundoNeuronioCamadaEntrada = round(
            np.tanh(np.sum(entradas * self.pesosSegundoNeuronioCamadaEntrada)), 6)

        self.saidaPrimeiroNeuronioCamadaOculta = round(
            np.tanh(np.sum(np.array([self.saidaPrimeiroNeuronioCamadaEntrada,
                                    self.saidaPrimeiroNeuronioCamadaEntrada]) * self.pesosPrimeiroNeuronioCamadaOculta)),
            6)

        self.saidaSegundoNeuronioCamadaOculta = round(
            np.tanh(np.sum(np.array([self.saidaPrimeiroNeuronioCamadaEntrada,
                                    self.saidaSegundoNeuronioCamadaEntrada]) * self.saidaSegundoNeuronioCamadaEntrada)),
            6)

        self.resultado = round(self.sigmoid(np.sum(np.array([self.saidaPrimeiroNeuronioCamadaOculta,
                                                            self.saidaSegundoNeuronioCamadaOculta]) * self.pesosNeuronioDeSaida)), 6)

        return self.resultado

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def atualizaPesos(self, erro, alpha=0.01):
        for i in range(len(self.pesosNeuronioDeSaida)):
            entrada = self.saidaPrimeiroNeuronioCamadaOculta if i == 0 else self.saidaSegundoNeuronioCamadaOculta
            self.pesosNeuronioDeSaida[i] += alpha * entrada * erro

        for i in range(len(self.pesosPrimeiroNeuronioCamadaOculta)):
            entrada1 = self.saidaPrimeiroNeuronioCamadaEntrada if i == 0 else self.saidaSegundoNeuronioCamadaEntrada
            self.pesosPrimeiroNeuronioCamadaOculta[i] += alpha * entrada1 * erro

        for i in range(len(self.pesosSegundoNeuronioCamadaOculta)):
            entrada2 = self.saidaPrimeiroNeuronioCamadaEntrada if i == 0 else self.saidaSegundoNeuronioCamadaEntrada
            self.pesosSegundoNeuronioCamadaOculta[i] += alpha * entrada2 * erro

        for i in range(len(self.pesosPrimeiroNeuronioCamadaEntrada)):
            self.pesosPrimeiroNeuronioCamadaEntrada[i] += alpha * erro

        for i in range(len(self.pesosSegundoNeuronioCamadaEntrada)):
            self.pesosSegundoNeuronioCamadaEntrada[i] += alpha * erro


class PongGame:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ping Pong")
        self.clock = pygame.time.Clock()

        self.green = (0, 200, 200)
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.font = pygame.font.Font(None, 30)

        self.rect_x = 272
        self.rect_y = 470
        self.x_cor = random.randint(15, width - 15)
        self.y_cor = random.randint(15, height - 15)
        self.x_change = random.randint(3, 7)
        self.y_change = random.randint(3, 7)

        self.floor_collision = False
        self.win = False
        self.sec = 0
        self.t = pygame.time.get_ticks()

        self.melhor_tempo = 0
        self.partidas_jogadas = 0

        # Inicializa rede neural
        self.rede = RedeNeural()
        self.melhores_pesos = [
            self.rede.pesosPrimeiroNeuronioCamadaEntrada.copy(),
            self.rede.pesosSegundoNeuronioCamadaEntrada.copy(),
            self.rede.pesosPrimeiroNeuronioCamadaOculta.copy(),
            self.rede.pesosSegundoNeuronioCamadaOculta.copy(),
            self.rede.pesosNeuronioDeSaida.copy()
        ]

        # Aqui criamos a lista para armazenar histórico (entradas, decisões)
        self.historico = []

    def move_player(self):
        decisao = self.rede.feedforward(self.rect_x, self.x_cor, self.y_cor)
        movimento = (decisao - 0.5) * 10  # movimento entre -5 e 5

        # Salvar entradas e decisão no histórico
        self.historico.append((self.rect_x, self.x_cor, self.y_cor, movimento))

        # Corrige movimento para evitar ficar preso nos cantos
        if self.rect_x <= 0 and movimento < 0:
            movimento = abs(movimento)
        elif self.rect_x >= self.width - 100 and movimento > 0:
            movimento = -abs(movimento)

        self.rect_x += movimento
        self.rect_x = max(0, min(self.rect_x, self.width - 100))

    def atualizar_bola(self):
        self.x_cor += self.x_change
        self.y_cor += self.y_change

        # Colisão com paredes laterais
        if self.x_cor > (self.width - 15) or self.x_cor < 15:
            self.x_change *= -1

        # Colisão com topo e chão
        if self.y_cor < 15:
            self.y_change *= -1

        # Colisão com a barra
        barra_rect = pygame.Rect(self.rect_x, self.rect_y, 100, 100)
        bola_rect = pygame.Rect(self.x_cor - 15, self.y_cor - 15, 30, 30)

        if bola_rect.colliderect(barra_rect):
            self.y_change *= -1

        # Bola passou da barra
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
        for entrada_raquete, entrada_bola_x, entrada_bola_y, movimento in self.historico:
            erro = (entrada_raquete + movimento - entrada_bola_y) / 100  # erro simples (ajuste conforme desejar)
            self.rede.feedforward(entrada_raquete, entrada_bola_x, entrada_bola_y)
            self.rede.atualizaPesos(erro)

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
