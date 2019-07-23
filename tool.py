import chainer
import chainer.functions as F
import cv2
import numpy as np
from chainer import Variable

labels_12 = ['MYSV', 'ZYMV', 'CCYV', 'CMV', 'PRSV', 'WMV', 'KGMMV',
             'HEALTHY', 'BrownSpot', 'DownyMildew', 'GrayMold', 'PowderyMildew']
label_dic_12 = {i: a for i, a in enumerate(labels_12)}


def generate(input_image, gen):

    x_in = np.asarray(input_image).astype(
        "f").transpose(2, 0, 1) / 128.0 - 1.0
    x_in = x_in.reshape(1, 3, 256, 256)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            x_in = Variable(x_in)
            x_out = gen(x_in)

    out = np.asarray(
        np.clip(x_out.array * 128 + 128, 0.0, 255.0), dtype=np.uint8)
    out = (out.reshape(3, 256, 256)).transpose(1, 2, 0)

    return out


def classifier(input_image, classify):

    x_in = input_image.astype("f")
    x_in = cv2.resize(x_in, (224, 224)).transpose(2, 0, 1) / 255.0
    x_in = x_in.reshape(1, 3, 224, 224)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            x_in = Variable(x_in)
            x_out = classify.predictor(x_in)
            probs = F.softmax(x_out).array

    top = np.argsort(probs)
    top = top.tolist()[0]
    labels = [label_dic_12[i] for i in top]
    results = [round(probs[0][i]*100, 3) for i in top]

    return labels[::-1], results[::-1]


def grad_cam(input_image, classify, logger=None):
    img = cv2.resize(input_image, (224, 224)).astype(
        "f").transpose(2, 0, 1) / 255.0
    img = img.reshape(1, 3, 224, 224)
    input_img = Variable(img)

    with chainer.using_config('train', False):
        pred = classify.predictor(input_img)
    probs = F.softmax(pred).data[0]
    top = np.argsort(probs)[::-1][0]

    pred.zerograd()
    pred.grad = np.zeros([1, 12], dtype=np.float32)
    pred.grad[0, top] = 1
    pred.backward(True)

    feature = classify.predictor.cam.data[0]
    grad = classify.predictor.cam.grad[0]

    cam = np.ones(feature.shape[1:], dtype=np.float32)
    weights = grad.mean((1, 2))*5000
    if logger:
        logger.info(f'Gradients: {weights}')

    for i, w in enumerate(weights):
        cam += feature[i] * w

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)
    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    image = img[0, :].transpose(1, 2, 0) * 255
    image -= np.min(image)
    image = cv2.cvtColor(np.minimum(image, 255), cv2.COLOR_BGR2RGB)
    cam_img = np.float32(heatmap) + np.float32(image)
    cam_img = 255 * cam_img / np.max(cam_img)

    return cam_img


def center_crop(img):
    height, width, channels = img.shape
    if width >= height:
        size = height
    else:
        size = width
    half_size = size // 2
    h = height // 2
    w = width // 2

    crop = img[h - half_size:h + half_size, w-half_size:w + half_size]
    return crop


def insert(con, inputname, disease, savename):
    cur = con.cursor()
    cur.execute('insert into results (inputname, disease, savename) values (?, ?, ?)', [
                inputname, disease, savename])
    pk = cur.lastrowid
    con.commit()
    return pk


def select(con, pk):
    cur = con.execute(
        'select id, inputname, disease, savename, created from results where id=?', (pk,))
    return cur.fetchone()


def select_all(con):
    cur = con.execute(
        'select id, inputname, disease, savename, created from results order by id desc')
    return cur.fetchall()


def delete(con, pk):
    cur = con.cursor()
    cur.execute('delete from results where id=?', (pk,))
    con.commit()
