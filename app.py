# Importing required libs
from flask import Flask, render_template, request
import os
import tensorflow as tf
import tensorflow_hub as hub


def predict_image_class(img_pa):
    model = tf.keras.models.load_model(("model.h5"), custom_objects={
                                       'KerasLayer': hub.KerasLayer})
    img = tf.keras.preprocessing.image.load_img(img_pa, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)  # Create a batch
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    predictions = model.predict(img)
    score = predictions.squeeze()
    if score < 0.29:
        print(f"This image is {100 * (1 - score):.2f}% benign.")
    elif score > 0.30 and score < 0.7:
        print("Got nothing")
    elif score > 0.71:
        print(f"This image is {100 * score:.2f}% malignant.")


# Instantiating flask app
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# Home route


@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                return render_template("result.html", err="Only .jpg .jpeg .png files are allowed")

            pred = predict_image_class(os.path.join(
                app.config['UPLOAD_FOLDER'], filename))
            return render_template("result.html", predictions=str(pred))

    except Exception as e:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


if __name__ == "__main__":
    app.run(port=9000, debug=True)
