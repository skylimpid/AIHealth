import numpy as np
from pandas import *
import datetime
import matplotlib

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from matplotlib.patches import Circle

import os


def generate_html_report(report_dir, patient_id, clean_img, bbox, predict):

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

    df = DataFrame(bbox, columns=['probablity', 'z-coordinator', 'y-coordinator', 'x-coordinator', 'nodule_diameter'])
    df.insert(0, 'nodule_id', list(range(1, len(bbox)+1)))
    df.index += 1

    now = datetime.datetime.now()
    now_time = now.strftime("%Y-%m-%d %H:%M")

    html = head.format(now_time, patient_id, predict)

    html += "<div>" + df.to_html() + "</div>"

    for index, row in df.iterrows():
        id = int(row['nodule_id'])
        z = int(row['z-coordinator'])
        y = row['y-coordinator']
        x = row['x-coordinator']
        diameter = row['nodule_diameter']

        circ = Circle((x, y), diameter, color='r', fill=False)

        fig, ax = plt.subplots(1)

        ax.set_aspect('equal')

        ax.imshow(clean_img[0][z])

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
    img = np.load('/home/xuan/AIHealthData/preprocess_result/ff8599dd7c1139be3bad5a0351ab749a_clean.npy')
    print(img.shape)
    generate_html_report("/home/xuan/AIHealthData/report", "patient_id", img, None, 0.5)