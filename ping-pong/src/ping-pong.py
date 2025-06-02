import pygame, sys, time, random
from pygame.locals import *
import math
import numpy as np

global pesos, bias, melhor_tempo, melhores_pesos, melhor_bias
global historico

pesos = np.random.uniform(-1, 1, size=5)
bias = np.random.uniform(-1, 1)
melhor_tempo = 0
melhores_pesos = pesos.copy()
melhor_bias = bias
historico = []
partidas_jogadas = 0

def neural_net(inputs, pesos, bias):
    # inputs e pesos são arrays NumPy
    soma = np.dot(inputs, pesos) + bias
    return np.tanh(soma)

def mutar(pesos, bias, sigma=0.1):
    novos_pesos = pesos + np.random.normal(0, sigma, pesos.shape)
    novo_bias = bias + np.random.normal(0, sigma)
    return novos_pesos, novo_bias

def treinar_com_historico(historico, pesos, bias, fitness, taxa=0.01):
    for inputs, decisao in historico:
        erro = fitness - decisao
        pesos += taxa * erro * inputs
        bias += taxa * erro
    return pesos, bias

# Função que inicializa o jogo com uma janela do tamanho (w, h)
def game_init(w,h):
    pygame.init()
    width,height = w,h
    size = width,height
    display = pygame.display.set_mode(size)
    pygame.display.set_caption("Ping Pong")
    return display

# Função atual que move o jogador com base no teclado (será substituída pela IA)
#ajeitar essa parte do código 
historico = []
def move_player(rect_x):
    inputs = np.array([x_cor, y_cor, x_change, y_change, rect_x])
    decisao = neural_net(inputs, pesos, bias)  # inclui bias
    historico.append((inputs.copy(), decisao))

    if decisao < -0.3:
        rect_x -= 5
    elif decisao > 0.3:
        rect_x += 5

    rect_x = max(0, min(rect_x, width - 100))
    return rect_x

def tela_fim(display, texto, font, text_color):
    display.fill((0,0,0))
    text_surface = font.render(texto, True, text_color)
    rect = text_surface.get_rect(center=(width//2, height//2 - 30))
    display.blit(text_surface, rect)

    instrucao = font.render("Pressione R para Reiniciar ou Q para Sair", True, text_color)
    rect2 = instrucao.get_rect(center=(width//2, height//2 + 30))
    display.blit(instrucao, rect2)

    pygame.display.flip()

# Dimensões da janela
width,height = 640, 480
display = game_init(width,height)
display_window = pygame.display.set_mode((width, height))

fps = 25  # frames por segundo

sec = 0  # contador de tempo
t = pygame.time.get_ticks()  # tempo inicial
clock = pygame.time.Clock()  # controle de FPS

# Cores
green = (0,200,200)
black = (0, 0, 0)
white = (255, 255, 255)

font = pygame.font.Font(None,30)
text_color = green

# Posição inicial da barra (player)
rect_x = 272
rect_y = 470

floor_collision = False  # controle de colisão com o chão
win = False              # controle de vitória

# Posição e velocidade inicial da bola
x_cor = random.randint(15, width - 15)
y_cor = random.randint(15, height - 15)
x_change = random.randint(3, 7)
y_change = random.randint(3, 7)

coordinates = []  # não está sendo usada no momento

music = pygame.mixer.Sound('musics/endofline.ogg')

# ============ LOOP PRINCIPAL ============

while True:
    # Eventos do teclado
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

    key = pygame.key.get_pressed()

    # Atualiza o contador de tempo
    if ((pygame.time.get_ticks()-t) >= 1000):
        sec += 1
        t = pygame.time.get_ticks()
        if (sec <= 60):
            if floor_collision == True:
                sec -= 1  # se perdeu, não conta tempo
            elif (sec >= 60):
                win = True

    music.play()  # toca a música

    rect_x = move_player(rect_x)
    if floor_collision:
        partidas_jogadas += 1

        fitness = sec  # fitness = tempo sobrevivido

        # Treina com dados da partida
        pesos, bias = treinar_com_historico(historico, pesos, bias, fitness)

        # Seleção simples
        if fitness >= melhor_tempo:
            melhor_tempo = fitness
            melhores_pesos = pesos.copy()
            melhor_bias = bias
        else:
            pesos = melhores_pesos.copy()
            bias = melhor_bias

        # A cada 10 partidas faz mutação para explorar
        if partidas_jogadas % 10 == 0:
            pesos, bias = mutar(pesos, bias, sigma=0.1)

        # Reseta variáveis para próximo jogo
        historico.clear()
        floor_collision = False
        sec = 0

    

    # Atualiza posição da bola
    x_cor += x_change
    y_cor += y_change

    # Limpa tela
    display_window.fill(black)

    # Desenha o tempo na tela
    time_text = font.render("Time: " + str(sec) + "s", True, text_color)
    display.blit(time_text,(10,10))

    # Desenha bolas passadas (sem uso prático no momento)
    for coordinate in coordinates:
        circle = pygame.draw.circle(display_window, white, (coordinate[0], coordinate[1]), 15, 0)

    # Desenha a bola atual
    circle = pygame.draw.circle(display_window, white, (x_cor, y_cor), 15, 0)

    # Desenha a barra do jogador
    rect = pygame.draw.rect(display, white, [rect_x, rect_y, 100, 100])

    # Tela de fim de jogo
    if floor_collision == True:
        display_window.fill(black)
        music.stop()
        text_fim = font.render("Game Over!", True, text_color)
        text_fim_rect = text_fim.get_rect()
        text_fim_rect.center = (display.get_width()//2, display.get_height()//2)
        display.blit(text_fim,text_fim_rect)
        esperando = True
        while esperando:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r: 
    # Só aceita a mutação se o jogo anterior foi melhor ou igual ao melhor até agora
                        if sec >= melhor_tempo:
                            melhor_tempo = sec
                            melhores_pesos = pesos
                            melhor_bias = bias
                        else:
                            pesos = melhores_pesos
                            bias = melhor_bias

                        # Aplica mutação
                        pesos, bias = mutar(pesos, bias, sigma=0.1)
                        floor_collision = False
                        sec = 0
                        t = pygame.time.get_ticks()

                        # resetar posição e velocidade da bola
                        x_cor = random.randint(15, width - 15)
                        y_cor = random.randint(15, height - 15)
                        x_change = random.randint(3, 7)
                        y_change = random.randint(3, 7)

                        # resetar posição da barra (opcional)
                        rect_x = 272

                        music.play()
                        esperando = False
                    elif event.key == pygame.K_q:  # sair
                        pygame.quit()
                        sys.exit()
            clock.tick(10)  # espera 10 FPS no loop de pausa
        continue  # volta para início do loop principal

    # Tela de vitória
    elif win == True:
        display_window.fill(black)
        music.stop()
        text_fim = font.render("You Win!", True, text_color)
        text_fim_rect = text_fim.get_rect()
        text_fim_rect.center = (display.get_width()//2, display.get_height()//2)
        display.blit(text_fim,text_fim_rect)

    # Colisão com paredes laterais
    if x_cor > (width - 15) or x_cor < 15:
        x_change = x_change * -1

    # Colisão com topo
    if y_cor > (height - 15) or y_cor < 15:
        y_change = y_change * -1

    # Colisão da bola com a barra
    if circle.colliderect(rect):
        y_change = y_change * -1

    # Bola passou da barra
    if y_cor > 460:
        floor_collision = True
        y_change = y_change * -1

    clock.tick(fps)
    # Mostrar tempos na tela
    # Mostrar tempos na tela, evitando sobreposição
    time_text = font.render(f"Tempo: {sec}s", True, text_color)
    best_text = font.render(f"Melhor tempo: {melhor_tempo}s", True, text_color)

    time_rect = time_text.get_rect(topleft=(10, 10))
    display_window.blit(time_text, time_rect)

    best_pos_y = time_rect.bottom + 5
    best_rect = best_text.get_rect(topleft=(10, best_pos_y))
    display_window.blit(best_text, best_rect)

    pygame.display.update()
    pygame.display.flip()
    time.sleep(0.015)
