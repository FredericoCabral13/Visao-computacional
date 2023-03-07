import struct
import six
import collections
import cv2
import datetime
from threading import Thread
from matplotlib import colors

class FPS:
    def __init__(self):
        # hora de início, hora de término, número total de frames
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # inicia o cronômetro
        self._start = datetime.datetime.now()
        return self
    
    def stop(self):
        # para o cronômetro
        self._end = datetime.datetime.now()

    def update(self):
        # incrementa o número de frames analisados ​​durante o intervalo inicial e final
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds
        return (self._end - self._start).total_seconds()

    def fps(self):
       # quadros aproximados por segundo
        return self._numFrames / self.elapsed()

    def get_numFrames(self):
        return self._numFrames


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # inicia o fluxo da câmera de vídeo
        # lê o primeiro quadro do stream

        self.stream = cv2.VideoCapture(src) # VideoCapture fornece API para capturar vídeo de câmeras, ler arquivos de vídeo e sequências de imagens
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # lê o primeiro quadro
        (self.grabbed, self.frame) = self.stream.read() # 'self.grabbed' recebe 'self.stream' e 'self.frame' recebe as dimensões width e height de 'self.stream' do frame lido mais recentemente
        self.stopped = False


    def start(self):
        # inicia o thread para ler os quadros do fluxo de vídeo
        Thread(target=self.update, args=()).start() #inicia-se o método 'update', atribuindo-o à variável 'target', e sem usar argumentos
        return self


    def update(self):
        # mantenha o loop até que o thread seja interrompido
        while True:
            if self.stopped: #se self.stopped == True
                return # não retorna nada

            # continue lendo o próximo quadro do stream de vídeo
            (self.grabbed, self.frame) = self.stream.read() # 'self.grabbed' recebe 'self.stream' e 'self.frame' recebe as dimensões width e height de 'self.stream' do frame lido mais recentemente


    def read(self):
        # obtém o frame lido mais recentemente
        return self.frame


    def stop(self):
        # parando a thread
        self.stopped = True
        self.stream.release() # interromper o thread (dando Crtl+C, por exemplo)


def standard_colors():
    colors = [
        'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
        'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
        'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
        'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
        'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
        'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
        'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
        'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
        'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
        'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
        'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
        'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
        'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
        'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
        'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
        'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
        'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
        'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
        'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
        'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
        'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
        'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
        'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]
    return colors

def color_name_to_rgb():
    colors_rgb = []
    for key, value in colors.cnames.items(): # para os títulos e valores que contém uma visualização das cores em hexadecimal
        colors_rgb.append((key, struct.unpack('BBB', bytes.fromhex(value.replace('#', ''))))) # adiciona à lista 'colors_rgb' as tuplas com o título e a conversão para hexadecimal (alterando os caracteres '#' para nada)
    return dict(colors_rgb)


def draw_boxes_and_labels(
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        keypoints=None,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False):
    """Retorna as coordenadas das caixas, nomes de classe e cores
    Argumentos:
      boxes: uma matriz numpy de forma [N, 4]
      classes: uma matriz numpy de forma [N]
      scores: uma matriz numpy de forma [N] ou None. Se pontuações=None, então
        esta função assume que as caixas a serem plotadas são groundtruth
        caixas e plotar todas as caixas como pretas sem classes ou pontuações.
      category_index: um dict contendo dicionários de categoria (cada
        índice de categoria `id` e nome de categoria `name`) codificados por índices de categoria.
      instance_masks: uma matriz numpy de forma [N, image_height, image_width], pode
        ser None.
      keypoints: uma matriz numpy de forma [N, num_keypoints, 2], pode
        ser nenhum
      max_boxes_to_draw: número máximo de caixas a serem visualizadas. Se nenhum, desenhe
        todas as caixas.
      min_score_thresh: limite mínimo de pontuação para uma caixa ser visualizada
      agnostic_mode: booleano (padrão: Falso) controlando se deve ser avaliado em
        modo class-agnóstico ou não. Este modo exibirá pontuações, mas ignorará
        classes.
    """
    # Crie uma string de exibição (e cor) para cada localização de caixa, agrupe todas as caixas
    # que correspondem ao mesmo local.
    box_to_display_str_map = collections.defaultdict(list) #faz o objeto 'list' virar tipo um dicionário e o atribui a uma variável
    box_to_color_map = collections.defaultdict(str) #faz o objeto 'str' virar tipo um dicionário e o atribui a uma variável
    box_to_instance_masks_map = {} #cria um dicionário vazio
    box_to_keypoints_map = collections.defaultdict(list) #faz o objeto 'list' virar tipo um dicionário e o atribui a uma variável
    if not max_boxes_to_draw: # se não tiver 'max_boxes_to_draw'
        max_boxes_to_draw = boxes.shape[0] # atribui 0 dimensões para 'boxes'
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh: # se 'scores' = None ou 'scores' > .5
            box = tuple(boxes[i].tolist()) # o array 'boxes' vira uma tupla com os mesmos elementos
            if instance_masks is not None: # se 'instance_masks' não for None
                box_to_instance_masks_map[box] = instance_masks[i] # a variável dicionário em que cada elemento da tupla 'box' significa cada elemento de 'instance_masks'
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i]) # extende a lista 'box_to_keypoints_map' da tupla 'box' aos elementos de 'keypoint'
            if scores is None:
                box_to_color_map[box] = 'black' # atribui a cor preta à tupla 'box'
            else:
                if not agnostic_mode: # se 'agnostic_mode' = True
                    if classes[i] in category_index.keys(): # se os elementos de 'classes' forem os títulos de 'category_index'
                        class_name = category_index[classes[i]]['name'] # 'class_name' passa a ser os nomes elementos de 'category_index' respectivo a cada coluna de 'classes'
                    else:
                        class_name = 'N/A' # se não houver elementos em 'class_name'
                    display_str = '{}: {}%'.format( # substituirá a primeira chave por 'class_name' e a segunda por 'int(100 * scores[i])'
                        class_name,
                        int(100 * scores[i]))
                else: # se 'agnostic_mode' = False
                    display_str = 'score: {}%'.format(int(100 * scores[i])) # substituirá a chave por 'int(100 * scores[i])'
                box_to_display_str_map[box].append(display_str) # o elemento 'box' em 'box_to_display_str_map' ganha o dicionário 'display_str'
                if agnostic_mode: # se 'agnostic_mode' == True
                    box_to_color_map[box] = 'DarkOrange' # ao elemento 'box' de 'box_to_color_map' é atribuído a cor DarkOrange
                else:
                    box_to_color_map[box] = standard_colors()[ #pega a função 'standard_colors' que é uma lista de nomes de cores
                        classes[i] % len(standard_colors())] # pega o elemento dessa lista que é o resto da divisão dos elementos de 'classes' pelo tamanho total da lista 'standard_colors'

    # Cópia da matriz, convertida para um tipo especificado.
    color_rgb = color_name_to_rgb() # atribui à variável a função que pega tem listas de tuplas de títulos e os valores das cores em hexadecimal
    rect_points = []
    class_names = []
    class_colors = []
    for box, color in six.iteritems(box_to_color_map): # para as variáveis 'box' e 'color' que contém pares de tuplas (chave, valor) de 'box_to_color_map'
        ymin, xmin, ymax, xmax = box # atribui cada um dos 4 valores de valores de 'box' a cada uma das variáveis, respectivamente
        rect_points.append(dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)) # adciona o dicionário à lista vazia 'rect_points'
        class_names.append(box_to_display_str_map[box]) # adiciona o elemento 'box' de 'box_to_display_str_map' à lista vazia 'class_names'
        class_colors.append(color_rgb[color.lower()]) # adiciona o elementos com o nome das cores todas em minúsculo à lista vazia 'class_colors'
    return rect_points, class_names, class_colors