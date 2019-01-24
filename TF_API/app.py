from flask import Flask, request, render_template, jsonify, abort, Response
import werkzeug
import os
import tf_api

# Create the application instance
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/upload_page')
def upload_page():
   return render_template('upload.html')

@app.route('/api/uploader', methods = ['POST'])
def upload():
    try:
        f = request.files['file']
    except:
        return Response("{'error' : 'Image missing'}", status = 400, mimetype='application/json')
    try:
        filename = os.path.join('uploads', werkzeug.secure_filename(f.filename))
        f.save(filename)
    except:
        return Response("{'error' : 'Internal error'}", status = 500, mimetype='application/json')
    try:
        result = jsonify(result = tf_api.infer_image(filename))
        os.remove(filename)
        return result
    except:
        os.remove(filename)
        return Response("{'error' : 'Unable to process image'}", status = 500, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)