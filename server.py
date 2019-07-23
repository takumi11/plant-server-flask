import os
import sqlite3
from datetime import datetime

import chainer
import chainer.links as L
import cv2
import numpy as np
import tool
from flask import Flask, g, render_template, request, send_from_directory
from model import classifier, re_attention_unet

ORG_DIR = "./images/origin"
MASKES_DIR = "./images/masked"
GRADCAM_DIR = "./images/gradcam"

if not os.path.isdir(GRADCAM_DIR):
    os.mkdir(GRADCAM_DIR)
if not os.path.isdir(MASKES_DIR):
    os.mkdir(MASKES_DIR)
if not os.path.isdir(ORG_DIR):
    os.mkdir(ORG_DIR)

print("load model")
gen = re_attention_unet.Generator(in_ch=3, out_ch=3, upsample='conv')
chainer.serializers.load_npz('./model/weight/re_attention_unet.npz', gen)
classify = L.Classifier(classifier.ResNet50_Fine(output=12))
chainer.serializers.load_npz('./model/weight/snap_model_12.npz', classify)
print("finish")

app = Flask(__name__, static_url_path="")
app.config.from_object(__name__)
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'db.sqlite3'),
    SECRET_KEY='foo-baa',
))


def connect_db():
    """ データベス接続に接続します """
    con = sqlite3.connect(app.config['DATABASE'])
    con.row_factory = sqlite3.Row
    return con


def get_db():
    """ connectionを取得します """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db


@app.teardown_appcontext
def close_db(error):
    """ db接続をcloseします """
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()


@app.route('/images/origin/<path:path>')
def send_origin(path):
    return send_from_directory(ORG_DIR, path)


@app.route('/images/masked/<path:path>')
def send_treated(path):
    return send_from_directory(MASKES_DIR, path)


@app.route('/images/gradcam/<path:path>')
def send_gradcam(path):
    return send_from_directory(GRADCAM_DIR, path)


@app.route('/static/leaf.jpg')
def send_jpg():
    return send_from_directory("./static", "leaf.jpg")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classification', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('classification.html')

    if request.method == 'POST':
        # 画像として読み込み
        file = request.files['file']
        file_name = file.filename
        stream = file.stream
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        origin_path = "images/origin/{}.jpg".format(now)
        masked_path = "images/masked/{}.jpg".format(now)
        gradcam_path = "images/gradcam/{}.jpg".format(now)

        # encode
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        try:
            org = cv2.imdecode(img_array, 1)

            org = tool.center_crop(org)
            org = cv2.resize(org, (256, 256))
            cv2.imwrite(origin_path, org)
            org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)

            out = tool.generate(org, gen)
            cv2.imwrite(masked_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            labels, results = tool.classifier(out, classify)
            gradcam = tool.grad_cam(out, classify, logger=app.logger)
            cv2.imwrite(gradcam_path, gradcam)

            con = get_db()
            pk = tool.insert(con, file_name, labels[0], now+".jpg")

            return render_template('classification.html',
                                   name_path=now + ".jpg",
                                   pk=pk, prob=labels[0],
                                   labels=labels, results=results)

        except Exception as e:
            app.logger.exception(e)
            return render_template('bad_request.html')


@app.route('/view/<pk>')
def view(pk):
    con = get_db()
    result = tool.select(con, pk)

    return render_template('view.html', result=result)


@app.route('/delete/<pk>', methods=['POST'])
def delete(pk):
    con = get_db()
    tool.delete(con, pk)

    return render_template('classification.html')


@app.route('/archive')
def archive():
    con = get_db()
    results = tool.select_all(con)
    return render_template('archive.html', results=results)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000, threaded=True)
