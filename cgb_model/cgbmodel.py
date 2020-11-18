"""
(c) 2020 Ruslan Guseinov (ruslan.guseinov@ist.ac.at), IST Austria
         Konstantinos Gavriil, TU Wien
Provided under MIT License

Example of MDN training and usage
"""
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="""Train and use cold glass bending MDN model
convert:
    --input-path  : data input directory (contains <optimal_panels_data.parq>)
    --output-path : data output directory (receives <*.npy> files)
train:
    --input-path : data directory (contains <optimal_panels_data.parq>)
    --model-path : model directory (contains <model_options.json>, receives <model.hdf5>)
predict:
    --input_path  : input Bezier surface
    --output-path : output Bezier directory
    --model-path  : model directory (contains <model.hdf5>)
    --output-name : output Bezier name

    N.B.: vertices in OBJ file representing Bezier control polygon have to be ordered row by row as follows:
     0  1  2  3
     4  5  6  7
     8  9 10 11
    12 13 14 15
    Internal control points (5, 6, 9, and 10) do not play any role in prediction but have to be provided with any value.
    Faces are ignored.
""")
parser.add_argument('command', choices=['convert', 'train', 'predict'])
parser.add_argument('-i', '--input-path', type=str, help='input file/directory')
parser.add_argument('-o', '--output-path', type=str, help='output file/directory')
parser.add_argument('-m', '--model-path', type=str, help='model directory')
parser.add_argument('-n', '--output-name', type=str, help='name output files')
parser.add_argument('-b', '--batch-size', default=500000, help='maximal batch size to avoid RAM overflow')
kargs = parser.parse_args()

import os.path as path
import numpy as np
import numpy.matlib
import pandas as pd
from panel_predictor import PanelPredictor
from cgb_utils import BEZIER_QUADS, BZ_COLS, permute_cp, cp2rep

command = kargs.command
input_path = kargs.input_path
output_path = kargs.output_path
model_path = kargs.model_path
output_name = kargs.output_name
b_size = kargs.batch_size

if input_path: input_path = input_path.rstrip('/\\')
if output_path: output_path = output_path.rstrip('/\\')
if model_path: model_path = model_path.rstrip('/\\')
if output_name: output_name = output_name.rstrip('/\\')

if command == 'convert':
    if not input_path: input_path = '../data'
    if not output_path: output_path = '../data'
elif command == 'train':
    if not input_path: input_path = '../data'
    if not model_path: model_path = '../data/mdn_model'
elif command == 'predict':
    if not input_path: input_path = '../data/bezier.obj'
    if not output_path: output_path = '../data/'
    if not model_path: model_path = '../data/mdn_model'
    if not output_name: output_name = 'bezier_prediction'

if command == 'convert':
    optimal_panels_file_name = 'optimal_panels_data.parq'
    db = pd.read_parquet(path.join(input_path, optimal_panels_file_name))
    dbt = {
        'train': db[db['dataset'] == 'train'],
        'test': db[db['dataset'] == 'test'],
    }
    for tt in ['train', 'test']:
        df = dbt[tt]
        cp = df[BZ_COLS].to_numpy().reshape(-1, 16, 3)
        cp = np.tile(cp, (2, 1, 1))
        cp[df.shape[0]:] *= -1
        stress = df['Lp12_stress'].to_numpy()
        stress = np.tile(stress, 2)
        reps = []
        for i in range((cp.shape[0] - 1) // b_size + 1):
            cpx = cp[i * b_size:(i + 1) * b_size]
            stressx = stress[i * b_size:(i + 1) * b_size]
            # To make output in the same order as legacy code
            cps = permute_cp(cpx)
            rep, _, _, _ = cp2rep(cps)
            rep = np.c_[rep, np.tile(stressx, (4, 1)).T.flatten()]
            reps.append(rep)
        reps = np.concatenate(reps)
        np.save(path.join(output_path, 'x_' + tt + '.npy'), reps[:,:18])
        np.save(path.join(output_path, 'y_' + tt + '.npy'), reps[:,18:])
if command == 'train':
    pp = PanelPredictor()
    pp.create(input_path, model_path)
    pp.train(model_path)
elif command == 'predict':
    with open(input_path, 'r') as obj_file:
        K = [[float(i) for i in line.split()[1:]] for line in obj_file.readlines() if line.startswith('v')]
    K = np.array(K)

    pp = PanelPredictor()
    pp.load(model_path)

    KK = np.reshape(np.matlib.repmat(K, 10, 1), (-1, 16, 3))
    pp.predict(KK)

    cp_pred, stress_pred, pis = pp.predict(K)

    for c in range(2):
        with open(path.join(output_path, output_name + f'_{c}.obj'), 'w') as obj_file:
            obj_file.writelines('v ' + ' '.join(str(i) for i in v) + '\n' for v in cp_pred[c])
            obj_file.writelines('f ' + ' '.join(str(i + 1) for i in f) + '\n' for f in BEZIER_QUADS)
    print('Lp12_stress predictions:', stress_pred * 1e-6, 'MPa')
    print('Component weights:', pis)

    # For multiple Bezier surfaces
    #KK = np.reshape(np.matlib.repmat(K, 10, 1), (-1, 16, 3))
    #pp.predict(KK)
