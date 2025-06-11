import pygame
import sys
import random
import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt
from random import uniform



class RedeNeural:
    def __init__(self):
        # Inicializa pesos da rede como atributos do objeto
        self.pesosPrimeiroNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(6)])
        self.pesosSegundoNeuronioCamadaEntrada = np.array([uniform(-1, 1) for _ in range(6)])

        self.pesosPrimeiroNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])
        self.pesosSegundoNeuronioCamadaOculta = np.array([uniform(-1, 1) for _ in range(2)])

        self.pesosNeuronioDeSaida = np.array([uniform(-1, 1) for _ in range(2)])

        self.resultado = 0

    def feedforward(self, XRaquete, XBolinha, YBola, VelocidadeX, VelocidadeY, bias=-1):
            entradas = np.array([XRaquete, XBolinha, YBola, VelocidadeX, VelocidadeY, bias])

            self.saidaPrimeiroNeuronioCamadaEntrada = np.tanh(np.sum(entradas * self.pesosPrimeiroNeuronioCamadaEntrada))
            self.saidaSegundoNeuronioCamadaEntrada = np.tanh(np.sum(entradas * self.pesosSegundoNeuronioCamadaEntrada))

            self.saidaPrimeiroNeuronioCamadaOculta = np.tanh(np.sum(np.array([
                self.saidaPrimeiroNeuronioCamadaEntrada,
                self.saidaSegundoNeuronioCamadaEntrada
            ]) * self.pesosPrimeiroNeuronioCamadaOculta))

            self.saidaSegundoNeuronioCamadaOculta = np.tanh(np.sum(np.array([
                self.saidaPrimeiroNeuronioCamadaEntrada,
                self.saidaSegundoNeuronioCamadaEntrada
            ]) * self.pesosSegundoNeuronioCamadaOculta))

            self.resultado = self.sigmoid(np.sum(np.array([
                self.saidaPrimeiroNeuronioCamadaOculta,
                self.saidaSegundoNeuronioCamadaOculta
            ]) * self.pesosNeuronioDeSaida))

            return self.resultado


    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Para evitar overflow
        return 1 / (1 + np.exp(-x))

    def atualizaPesos(self, erro, entradas, alpha=0.58):
        # Atualiza os pesos através de gradiente descendente
        for i in range(len(self.pesosNeuronioDeSaida)):
            entrada_oculta = self.saidaPrimeiroNeuronioCamadaOculta if i == 0 else self.saidaSegundoNeuronioCamadaOculta
            self.pesosNeuronioDeSaida[i] += alpha * entrada_oculta * erro

        for i in range(len(self.pesosPrimeiroNeuronioCamadaOculta)):
            entrada_entrada = self.saidaPrimeiroNeuronioCamadaEntrada if i == 0 else self.saidaSegundoNeuronioCamadaEntrada
            self.pesosPrimeiroNeuronioCamadaOculta[i] += alpha * entrada_entrada * erro

        for i in range(len(self.pesosSegundoNeuronioCamadaOculta)):
            entrada_entrada = self.saidaPrimeiroNeuronioCamadaEntrada if i == 0 else self.saidaSegundoNeuronioCamadaEntrada
            self.pesosSegundoNeuronioCamadaOculta[i] += alpha * entrada_entrada * erro

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
                    if len(row) == 7:
                        entrada = list(map(float, row[:6]))
                        decisao = float(row[6])
                        historico.append({'entrada': entrada, 'decisao': decisao})
        return historico

    def plotar_graficos(self):
        print("Plotando gráficos...")  # Adiciona log para depuração

        # Gráfico de erro ao longo do tempo
        if len(self.erros) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(self.erros)
            plt.title("Evolução do Erro")
            plt.xlabel("Epoch")
            plt.ylabel("Erro")
            plt.grid(True)
            plt.savefig("grafico_erro.png")  # Salva o gráfico de erro
            plt.show()  # Exibe o gráfico imediatamente
            plt.close()

        # Gráfico de pesos da rede neural ao longo do tempo
        if len(self.pesos_hist) > 0:
            pesos_media = [np.mean(np.array(pesos)) for pesos in self.pesos_hist]
            plt.figure(figsize=(10, 5))
            plt.plot(pesos_media)
            plt.title("Evolução dos Pesos")
            plt.xlabel("Epoch")
            plt.ylabel("Média dos Pesos")
            plt.grid(True)
            plt.savefig("grafico_pesos.png")  # Salva o gráfico de pesos
            plt.show()  # Exibe o gráfico imediatamente
            plt.close()

    def salvar_historico_csv(self):
        filename = f"historico_{self.id_thread}.csv" if self.headless else "historico.csv"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for registro in self.historico:
                linha = registro['entrada'] + [registro['decisao']]
                writer.writerow(linha)

    def move_player(self):
        # Normaliza as entradas
        XRaquete_norm = self.rect_x / self.width
        XBolinha_norm = self.x_cor / self.width
        YBola_norm = self.y_cor / self.height
        VelocidadeX_norm = self.x_change / 10
        VelocidadeY_norm = self.y_change / 10

        # Alimenta a rede com o X da raquete
        decisao = self.rede.feedforward(
            XRaquete_norm,
            XBolinha_norm,
            YBola_norm,
            VelocidadeX_norm,
            VelocidadeY_norm
        )

        # Movimento entre -5 e 5, ajustando proporcionalmente à velocidade da bolinha
        movimento = (decisao - 0.5) * 10  # Movimento entre -5 e 5

        # Ajuste da velocidade da raquete com base na velocidade da bolinha
        fator_proporcional = abs(VelocidadeX_norm) + abs(VelocidadeY_norm)  # A velocidade total da bolinha
        movimento *= fator_proporcional  # Proporcionalidade do movimento da raquete

        # Para garantir que a raquete tenha um movimento mínimo e não fique lenta
        if abs(movimento) < 1:
            movimento = 2 if movimento > 0 else -2

        # Ajustando o movimento para garantir que a raquete se mova para a borda correta
        if self.rect_x <= 1:
            self.frames_no_canto += 1
            movimento = 5  # Valor fixo para direita
        elif self.rect_x >= self.width - 101:
            self.frames_no_canto += 1
            movimento = -5  # Valor fixo para esquerda
        else:
            self.frames_no_canto = 0

        # Garante que a raquete não ultrapasse os limites da tela
        if self.rect_x + movimento < 0:
            movimento = -self.rect_x  # Garante que a raquete não ultrapasse a borda esquerda
        elif self.rect_x + movimento > self.width - 100:
            movimento = (self.width - 100) - self.rect_x  # Garante que a raquete não ultrapasse a borda direita

        self.rect_x += movimento  # Aplica o movimento na posição

        # Armazenar o movimento atual para evitar repetições no próximo ciclo
        self.ultimo_movimento = movimento

        entrada = [self.rect_x / self.width, self.x_cor / self.width, self.y_cor / self.height,
        self.x_change / 10, self.y_change / 10, -1]


    def atualizar_bola(self):
        self.x_cor += self.x_change
        self.y_cor += self.y_change

        # Impede que a bola saia dos limites da tela
        self.x_cor = max(15, min(self.x_cor, self.width - 15))
        self.y_cor = max(15, min(self.y_cor, self.height - 15))

        # Colisão com as paredes laterais
        if self.x_cor == 15 or self.x_cor == self.width - 15:
            self.x_change *= -1

        # Colisão com o topo (parte superior da tela)
        if self.y_cor == 15:
            self.y_change *= -1  # Inverte a direção da bola quando bate no topo

        # Colisão com a raquete
        barra_rect = pygame.Rect(self.rect_x, self.rect_y, 100, 100)
        bola_rect = pygame.Rect(int(self.x_cor - 15), int(self.y_cor - 15), 30, 30)

        if bola_rect.colliderect(barra_rect):
            self.y_change *= -1  # Inverte a direção da bola ao colidir com a raquete
            movimento_extra = 2.0  # Ajuste a velocidade para tornar o jogo mais dinâmico
            if self.rect_x < self.width / 2:  # Se a raquete estiver à esquerda
                self.rect_x += movimento_extra
            else:  # Se a raquete estiver à direita
                self.rect_x -= movimento_extra

        # Colisão com o chão
        if self.y_cor > self.height - 20:
            self.floor_collision = True
            self.y_change *= -1  # Reverte a direção quando atinge o chão
            self.reiniciar_jogo()  # Reinicia o jogo

    def reiniciar_jogo(self):
        """Reinicia o jogo, resetando variáveis"""
        # Definir a posição inicial da bola
        self.x_cor = random.randint(15, self.width - 15)  # Posição X aleatória
        self.y_cor = 15  # Posição Y no topo da tela (parte mais alta)
        
        # Inicializa a direção da bola com valores aleatórios para que a bola não fique estática
        self.x_change = random.randint(3, 7) * random.choice([-1, 1])  # Direção aleatória no eixo X
        self.y_change = random.randint(3, 7)  # Movimento sempre para baixo no eixo Y
        
        # Posição da raquete
        self.rect_x = 272
        self.rect_y = 470
        self.frames_no_canto = 0
        self.floor_collision = False
        self.sec = 0
        self.historico.clear()  # Limpar o histórico para o próximo treinamento

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
            penalidade_canto = 0
            if self.frames_no_canto > 1:
                penalidade_canto = 1.0  # penalidade forte para sair do canto

            # Calculando o erro incluindo a penalidade
            erro = (entradas[0] - entradas[1]) / 100 + penalidade_canto


        # Salvar erros e pesos para gráficos
        self.rede.feedforward(*entradas)
        self.rede.atualizaPesos(erro, entradas)

        # Armazenando erros e pesos para os gráficos
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

            pygame.display.flip()
            self.clock.tick(25)

        # Plotar gráficos quando o jogo terminar
        self.plotar_graficos()

        pygame.quit()
        sys.exit()


# Função que reinicia o jogo e a rede neural
def treinar_rede_multiplicadas_vezes(numero_de_treinamentos=10):
    for i in range(numero_de_treinamentos):
        print(f"Treinamento {i+1}/{numero_de_treinamentos}...")
        
        # Inicializa o jogo e a rede neural
        game = PongGame()  # Crie ou utilize a classe do seu jogo
        game.loop_principal()  # Roda o jogo, treinando e atualizando os pesos durante a execução

        # Aqui você pode adicionar condições para salvar os melhores desempenhos, se necessário
        # Exemplo: Salve o erro ou pesos para cada execução
        print(f"Treinamento {i+1} finalizado. Pesos atualizados!")
        
# Chame a função para treinar múltiplas vezes
treinar_rede_multiplicadas_vezes(5)  # Treinar 5 vezes, por exemplo


if __name__ == "__main__":
    treinar_rede_multiplicadas_vezes(5)  # Executa 5 treinamentos em paralelo
