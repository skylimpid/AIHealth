import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pandas import *
from Training.constants import DATA_BASE_DIR


def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates

def generate_html_report(report_dir, patient_id, clean_img, bbox, predict, spacing, origin):

    patient_report_dir = os.path.join(report_dir, patient_id)
    if not os.path.exists(patient_report_dir):
        os.makedirs(patient_report_dir, exist_ok=True)

    img_save_dir = os.path.join(patient_report_dir, "img")
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)

    head = '''
    <!DOCTYPE html>
    <html>
    <head>
      <center><h1>Diagnosis Report</h1></center>
      <center><p>{}</p></center>
    </head>

    <body>
    <h2>Patient Name: {}</h2>
    <h3>The probabilty to have cancer: {}</h3>
    <h3>The potential nodule position:</h3>
    
    '''

    end = '''
    </body>
    </html>
    '''

    img_div = '''
    <div>
    <h3> Snapshot for {}:</h3>
    <img src="{}" alt="{}" width="500" height="377">
    </div>
    '''

    # extend bbox to have world position
    bbox_extend = np.zeros((bbox.shape[0], bbox.shape[1] + 3))
    bbox_extend[:, 0:bbox.shape[1]] = bbox
    bbox_extend[:, bbox.shape[1]:] = voxel_2_world(bbox[:, 1:4], origin, spacing)

    df = DataFrame(bbox_extend, columns=['probablity','z-coordinator(voxel)','y-coordinator(voxel)',
                                         'x-coordinator(voxel)', 'nodule_diameter', 'z-coordinator(world)',
                                         'y-coordinator(world)', 'x-coordinator(world)'])
    df.insert(0, 'nodule_id', list(range(1, len(bbox)+1)))
    df.index += 1

    now = datetime.datetime.now()
    now_time = now.strftime("%Y-%m-%d %H:%M")

    html = head.format(now_time, patient_id, predict)

    html += "<div>" + df.to_html() + "</div>"

    for index, row in df.iterrows():
        id = int(row['nodule_id'])
        z = int(row['z-coordinator(voxel)'])
        y = row['y-coordinator(voxel)']
        x = row['x-coordinator(voxel)']
        diameter = row['nodule_diameter']

        circ = Circle((x, y), diameter, color='r', fill=False)

        fig, ax = plt.subplots(1)

        ax.set_aspect('equal')

        ax.imshow(clean_img[0][z], cmap='gray')

        ax.add_patch(circ)

        png_file = "{}.png".format(id)
        saved_png = os.path.join(img_save_dir, png_file)
        plt.savefig(saved_png)
        html += img_div.format("nodule_{}".format(id),saved_png, "nodule_{}".format(id))



    html += end

    report = os.path.join(patient_report_dir, "report.html")
    fh = open(report, "w")
    fh.write(html)
    fh.close()


if __name__ == "__main__":
    img = np.load(
        os.path.join(DATA_BASE_DIR + '/preprocess_result/0a0c32c9e08cc2ea76a71649de56be6d_clean.npy'))
    print(img.shape)
    bbox = np.load(
        os.path.join(DATA_BASE_DIR + '/classifier_intermediate/candidate_box/0a0c32c9e08cc2ea76a71649de56be6d_candidate.npy'))
    bbox = bbox[:2, :]
    print(bbox.shape)
    #print(bbox.shape)
    #print(bbox)
    generate_html_report(os.path.join(DATA_BASE_DIR + '/report'), "patient_id", img, bbox, 0.5, [0.5,0.25,0.3], [-120, -322, -145])

