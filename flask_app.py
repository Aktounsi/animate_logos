import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from src.pipeline import *

UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'svg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            logo = Logo(data_dir=os.path.join(app.config['UPLOAD_FOLDER'], filename))
            logo.animate(nb_animations=10)
            animation_filename_1 = filename.replace(".svg", "_animated_" + str(0) + ".svg")
            animation_filename_2 = filename.replace(".svg", "_animated_" + str(1) + ".svg")
            animation_filename_3 = filename.replace(".svg", "_animated_" + str(2) + ".svg")
            animation_filename_4 = filename.replace(".svg", "_animated_" + str(3) + ".svg")
            animation_filename_5 = filename.replace(".svg", "_animated_" + str(4) + ".svg")
            animation_filename_6 = filename.replace(".svg", "_animated_" + str(5) + ".svg")
            animation_filename_7 = filename.replace(".svg", "_animated_" + str(6) + ".svg")
            animation_filename_8 = filename.replace(".svg", "_animated_" + str(7) + ".svg")
            animation_filename_9 = filename.replace(".svg", "_animated_" + str(8) + ".svg")
            animation_filename_10 = filename.replace(".svg", "_animated_" + str(9) + ".svg")
            return render_template('show.html', filename=filename,
                                   animation_filename_1=animation_filename_1,
                                   animation_filename_2=animation_filename_2,
                                   animation_filename_3=animation_filename_3,
                                   animation_filename_4=animation_filename_4,
                                   animation_filename_5=animation_filename_5,
                                   animation_filename_6=animation_filename_6,
                                   animation_filename_7=animation_filename_7,
                                   animation_filename_8=animation_filename_8,
                                   animation_filename_9=animation_filename_9,
                                   animation_filename_10=animation_filename_10)
    return render_template('index.html')


@app.route('/data/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/data/uploads/<animation_filename_1>')
def animated_svg_1(animation_filename_1):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_1)


@app.route('/data/uploads/<animation_filename_2>')
def animated_svg_2(animation_filename_2):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_2)


@app.route('/data/uploads/<animation_filename_3>')
def animated_svg_3(animation_filename_3):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_3)


@app.route('/data/uploads/<animation_filename_4>')
def animated_svg_4(animation_filename_4):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_4)


@app.route('/data/uploads/<animation_filename_5>')
def animated_svg_5(animation_filename_5):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_5)


@app.route('/data/uploads/<animation_filename_6>')
def animated_svg_6(animation_filename_6):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_6)


@app.route('/data/uploads/<animation_filename_7>')
def animated_svg_7(animation_filename_7):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_7)


@app.route('/data/uploads/<animation_filename_8>')
def animated_svg_8(animation_filename_8):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_8)


@app.route('/data/uploads/<animation_filename_9>')
def animated_svg_9(animation_filename_9):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_9)


@app.route('/data/uploads/<animation_filename_10>')
def animated_svg_10(animation_filename_10):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_10)


@app.route('/redirect_button')
def redirect_button():
    return redirect(url_for('index', _anchor='upload_svg_logo'))


if __name__ == '__index__':
    app.debug = True
    app.run()
