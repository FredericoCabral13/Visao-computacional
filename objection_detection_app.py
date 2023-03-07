import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf

from utils.webcam import FPS, WebcamVideoStream
from queue import Queue
from threading import Thread
from analytics.tracking import ObjectTracker
from video_writer import VideoWriter
from detect_object import detect_objects

CWD_PATH = os.getcwd()

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' 
PATH_TO_MODEL = os.path.join(CWD_PATH, 'detection', MODEL_NAME, 'frozen_inference_graph.pb')
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#PATH_TO_MODEL = os.path.join(CWD_PATH, 'detection', 'tf_models', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_VIDEO = os.path.join(CWD_PATH, 'input.mp4')


def worker(input_q, output_q):
    # carrega o modelo tensorflow congelado na memória
    detection_graph = tf.Graph()
    with detection_graph.as_default(): #para criar vários gráficos simultaneamente, em que todas as operações abaixo são adicionadas ao gráfico padrão, ao invés de se criarem vários gráficos
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.io.gfile.GFile(PATH_TO_MODEL, 'rb') as fid: #acessando o modelo
            serialized_graph = fid.read() #lendo o modelo
            od_graph_def.ParseFromString(serialized_graph) # analisa a manesagem da string 'serialized_graph' fornecida
            tf.import_graph_def(od_graph_def, name='') #Importa o gráfico de graph_def para o Graph padrão atual.

        sess = tf.compat.v1.Session(graph=detection_graph) # o gráfico 'detection_graph' será usado para criar as sessões contendo cada gráficos

    fps = FPS().start() # dá início à contagem de frames
    while True:
        fps.update() # incrementa o número de frames analisados ​​durante o intervalo inicial e final
        frame = input_q.get() # a variável 'frame' é a fila de ítens como explicado lá embaixo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converte a imagem 'frame' de entrada de um espaço de cor para outro. 
        output_q.put(detect_objects(frame, sess, detection_graph)) 
        '''
        retorna a lista dos valores de x e y (min e max), 
        também a lista que contém a tupla de 'boxes', em que a cada elemento desses é atribuído um dicionário onde o nome da categoria, se existir, é atribuiído a um score,
        também a lista com o nome das cores
        '''
    fps.stop()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.') #adiciona um argumento inteiro para o 'parser' de nome 'video_source', no qual diz respeito que o vídeo vem da câmera
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=1280, help='Width of the frames in the video stream.') # // nome 'width' de valor padrão 1280 // largura dos frames no stream de vídeo
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=720, help='Height of the frames in the video stream.') #// nome 'height' de valor padrão 720 // altura dos frames no stream de vídeo
    args = parser.parse_args() # atribui à variável 'args' o objeto 'parser' como tendo todos os argumentos adicionados anteriormente

    input_q = Queue(5) #introduz à variável input_q uma fila de 5 elementos, no qual o primeiro a entrar é o primeiro a sair.
    output_q = Queue() #introduz à variável ouput_q uma fila sem elementos
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q)) #atribui à variável a função 'worker' com os parâmetros input_q e output_q
        t.daemon = True #atribui True para rodar t em segundo plano
        t.start() #inicia-se o processo de rodar t

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start() #ininia-se a captura de vídeo pela câmera e pegas as dimensões em que esta captura ocorrerá
    writer = VideoWriter('output.mp4', (args.width, args.height)) #pega o vídeo no formato mp4

    '''
    stream = cv2.VideoCapture(0)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    '''

    fps = FPS().start() #inicia-se a contagem de frames por segundo
    object_tracker = ObjectTracker(path='./', file_name='report.csv')
    while True:
        frame = video_capture.read()
        # (ret, frame) = stream.read()
        fps.update()

        if fps.get_numFrames() % 2 != 0:
            continue

        # put data into the input queue 
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            data = output_q.get()
            context = {'frame': frame, 'class_names': data['class_names'], 'rec_points': data['rect_points'], 'class_colors': data['class_colors'],
                        'width': args.width, 'height': args.height, 'frame_number': fps.get_numFrames()}
            new_frame = object_tracker(context)
            writer(new_frame)
            cv2.imshow('Video', new_frame)

        #print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    writer.close()
    cv2.destroyAllWindows()
