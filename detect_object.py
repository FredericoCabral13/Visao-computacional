import os
import numpy as np
from utils import label_map_util
from utils.webcam import draw_boxes_and_labels


CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH, 'detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90
# label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# passa na image_np, retorna
def detect_objects(image_np, sess, detection_graph):

    image_np_expanded = np.expand_dims(image_np, axis=0) # Expande a forma de uma matriz e insire um novo eixo que aparecerá na posição do eixo na forma de matriz expandida
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') # do gráfico 'detection_graph' pegar tensor pelo nome 'image_tensor:0' e atribuir à variável

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Faça a detecção/previsão do modelo aqui
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}) # a cada variável é atribuída uma sessão na qual pega cada variável dessas, que pegam um tensor pelo nome no gráfico padrão, e os alimenta com a imagem expandida 'image_np_expanded'
    
    rect_points, class_names, class_colors = draw_boxes_and_labels( # atribui cada um dos 3 elementos que a função retorna à cada variável, respectivamente
        boxes=np.squeeze(boxes), #remove os eixos de comprimento 1 de 'boxes' (matriz numpy de forma [N, 4])
        classes=np.squeeze(classes).astype(np.int32), # /// 'classes' (matriz numpy de forma [N]) e faz uma cópia da matriz, convertida para o tipo int32.
        scores=np.squeeze(scores),  # /// 'scores' (matriz numpy de forma [N] ou None. Se scores=None, então esta função assume que as caixas a serem plotadas são caixas de verdade fundamental e plota todas as caixas como pretas sem classes ou scores.)
        category_index=category_index, # 'category_index' um dicionário contendo dicionários de categoria (cada índice de categoria id e nome de categoria name) codificados por índices de categoria.
        min_score_thresh=.5 # limite mínimo de pontuação para uma caixa ser visualizada 
    )

    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)