import os
import sys

from flask import Flask,request,render_template,jsonify
from flask_cors import CORS

import torch
from torch.utils.data import DataLoader
import cv2

from werkzeug.utils import secure_filename
from matplotlib.ticker import NullLocator



root = os.path.split(os.path.abspath(__file__))[0]
ml_root = root+'/ml'
sys.path.append(ml_root)

try:
    from ml.models.models import *
    from ml.utils.utils import *
    from ml.utils.datasets import *
except:
    pass


root = os.path.split(os.path.abspath(__file__))[0]

dataset = 'fly'
model_def = os.path.join(root,'ml','config','fly','yolov3-custom-fly.cfg')
weights_path = os.path.join(root,'ml','weights','yolov3_ckpt_fly.pth')
class_path = os.path.join(root,'ml','data','custom','fly','classes.names')
image_folder = os.path.join(root,'static','images','test_images')
img_size = 416
batch_size = 1
n_cpu = 1
conf_thres = 0.8
nms_thres = 0.4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.join(root,'static','images','result_images'), exist_ok=True)
model = Darknet(model_def, img_size=img_size).to(device)
model.load_state_dict(torch.load(
    weights_path, map_location=torch.device(device)))
model.eval()
classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


ALLOWED_EXTENSIONS = set(['jpg', 'png', 'jpeg'])
UPLOAD_FOLDER = os.path.join(root,'static','images','test_images')

app = Flask(__name__, template_folder='template')
cors = CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'PrinceAPI'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            for i in os.listdir(UPLOAD_FOLDER):
                os.remove(UPLOAD_FOLDER+'/'+i)
        except:
            pass
        # check if the post request has the file part

        if 'files[]' not in request.files:
            resp = jsonify({'message': 'No file part in the request'})
            resp.status_code = 400

        files = request.files.getlist('files[]')

        errors = {}
        success = False
        file = files[0]
        filename = ""

        if file and allowed_file(file.filename):
            ts = time.time()
            filename = f"{str(ts)}-{file.filename}"
            filename = secure_filename(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors['message'] = 'File type is not allowed'

        if success and errors:
            resp = jsonify({"filepath": f"{app.config['UPLOAD_FOLDER']}/{filename}", "filename": filename,
                            'message': 'Files successfully uploaded'})
            resp.status_code = 206
        if success:
            resp = jsonify({"filepath": f"{app.config['UPLOAD_FOLDER']}/{filename}", "filename": filename,
                            'message': 'Files successfully uploaded'})
            resp.status_code = 201
        else:
            resp = jsonify(errors)
            resp.status_code = 400
            
        return {'html': render_template('home.html'), 'status': 200}
    return render_template('home.html')


@app.route('/process')
def process():
    value = None
    try:
        result_folder = os.path.join(root,'static','images','result_images')
        for i in os.listdir(result_folder):
            os.remove(os.path.join(result_folder,i))
    except:
        pass
    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )
    imgs = []
    img_detections = []

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = input_imgs[:, :3, :, :]
        # Configure input
        try:
            input_imgs = Variable(input_imgs.type(Tensor))
        except:
            pass
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(
                detections, conf_thres, nms_thres)

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Create plot
        img = np.array(Image.open(path))
        height,width,_ = img.shape

        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                box_w = x2 - x1
                box_h = y2 - y1

                color =bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                color =  (127/255, 0, 1,1)
                print(color)
                # Create a Rectangle patch
                bbox = patches.Rectangle(
                    (x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)], color="white", verticalalignment="top", bbox={
                             "color": color, "pad": 0},)

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1]
        
        plt.savefig(os.path.join(root,'static','images','result_images',filename), bbox_inches="tight", pad_inches=0.0)
        plt.close()

        inference_image = cv2.imread(os.path.join(root,'static','images','result_images',filename))
        inference_image = cv2.resize(inference_image,(width,height))
        cv2.imwrite(os.path.join(root,'static','images','result_images',filename),inference_image)

        value = filename
    return render_template('home.html', value=value)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
