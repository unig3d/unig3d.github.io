# modify from https://github.com/Xharlie/DISN/blob/master/preprocessing/create_point_sdf_grid.py

import h5py
import os
import argparse
import numpy as np

import trimesh
from scipy.interpolate import RegularGridInterpolator
import time

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='model path')
parser.add_argument('--reduce', type=int, default=1, help='define resolution. res=256//reduce')
parser.add_argument('--save_path', type=str, default='./tmp')
FLAGS = parser.parse_args()


def get_sdf_value(sdf_pt, sdf_params_ph, sdf_ph, sdf_res):
    x = np.linspace(sdf_params_ph[0], sdf_params_ph[3], num=sdf_res)
    y = np.linspace(sdf_params_ph[1], sdf_params_ph[4], num=sdf_res)
    z = np.linspace(sdf_params_ph[2], sdf_params_ph[5], num=sdf_res)
    my_interpolating_function = RegularGridInterpolator((z, y, x), sdf_ph)
    sdf_value = my_interpolating_function(sdf_pt)
    print("sdf_value:", sdf_value.shape)
    return np.expand_dims(sdf_value, axis=1)


def get_sdf(sdf_file, sdf_res):
    intsize = 4
    floatsize = 8
    sdf = {
        "param": [],
        "value": []
    }
    with open(sdf_file, "rb") as f:
        try:
            bytes = f.read()
            ress = np.fromstring(bytes[:intsize * 3], dtype=np.int32)
            if -1 * ress[0] != sdf_res or ress[1] != sdf_res or ress[2] != sdf_res:
                raise Exception(sdf_file, "res not consistent with ", str(sdf_res))
            positions = np.fromstring(bytes[intsize * 3:intsize * 3 + floatsize * 6], dtype=np.float64)
            # bottom left corner, x,y,z and top right corner, x, y, z
            sdf["param"] = [positions[0], positions[1], positions[2],
                            positions[3], positions[4], positions[5]]
            sdf["param"] = np.float32(sdf["param"])
            sdf["value"] = np.fromstring(bytes[intsize * 3 + floatsize * 6:], dtype=np.float32)
            sdf["value"] = np.reshape(sdf["value"], (sdf_res + 1, sdf_res + 1,
                                                     sdf_res + 1))  # somehow the cube is sdf_res+1 rather than sdf_res... need to investigate why
        finally:
            f.close()
    return sdf


def get_offset_ball(num, bandwidth):
    u = np.random.normal(0, 1, size=(num, 1))
    v = np.random.normal(0, 1, size=(num, 1))
    w = np.random.normal(0, 1, size=(num, 1))
    r = np.random.uniform(0, 1, size=(num, 1)) ** (1. / 3) * bandwidth
    norm = np.linalg.norm(np.concatenate([u, v, w], axis=1), axis=1, keepdims=1)
    (x, y, z) = r * (u, v, w) / norm
    return np.concatenate([x, y, z], axis=1)


def get_offset_cube(num, bandwidth):
    u = np.random.normal(0, 1, size=(num, 1))
    v = np.random.normal(0, 1, size=(num, 1))
    w = np.random.normal(0, 1, size=(num, 1))
    r = np.random.uniform(0, 1, size=(num, 1)) ** (1. / 3) * bandwidth
    norm = np.linalg.norm(np.concatenate([u, v, w], axis=1), axis=1, keepdims=1)
    (x, y, z) = r * (u, v, w) / norm
    return np.concatenate([x, y, z], axis=1)


def sample_sdf(sdf_dict, sdf_res, reduce):
    start = time.time()
    params = sdf_dict["param"]
    sdf_values = sdf_dict["value"].flatten()

    n_sample = sdf_res // reduce  # want 64 * 64 * 64

    x = np.linspace(params[0], params[3], num=n_sample).astype(np.float32)
    y = np.linspace(params[1], params[4], num=n_sample).astype(np.float32)
    z = np.linspace(params[2], params[5], num=n_sample).astype(np.float32)
    z_vals, y_vals, x_vals = np.meshgrid(z, y, x, indexing='ij')
    print("x_vals", x_vals[0, 0, sdf_res // reduce - 1])

    x_original = np.linspace(params[0], params[3], num=sdf_res + 1).astype(np.float32)
    y_original = np.linspace(params[1], params[4], num=sdf_res + 1).astype(np.float32)
    z_original = np.linspace(params[2], params[5], num=sdf_res + 1).astype(np.float32)
    x_ind = np.arange(n_sample).astype(np.int32)
    y_ind = np.arange(n_sample).astype(np.int32)
    z_ind = np.arange(n_sample).astype(np.int32)
    zv, yv, xv = np.meshgrid(z_ind, y_ind, x_ind, indexing='ij')
    choosen_ind = xv * reduce + yv * (sdf_res + 1) * reduce + zv * (sdf_res + 1) ** 2 * reduce
    choosen_ind = np.asarray(choosen_ind, dtype=np.int32).reshape(-1)
    vals = sdf_values[choosen_ind]

    sdf_pt_val = np.expand_dims(vals, axis=-1)
    print("sdf_pt_val.shape", sdf_pt_val.shape)
    print("sample_sdf: {} s".format(time.time() - start))
    return sdf_pt_val, check_insideout(sdf_values, sdf_res, x_original, y_original, z_original)


def check_insideout(sdf_val, sdf_res, x, y, z):
    # "chair": "03001627",
    # "bench": "02828884",
    # "cabinet": "02933112",
    # "car": "02958343",
    # "airplane": "02691156",
    # "display": "03211117",
    # "lamp": "03636649",
    # "speaker": "03691459",
    # "rifle": "04090263",
    # "sofa": "04256520",
    # "table": "04379243",
    # "phone": "04401088",
    # "watercraft": "04530566"
    x_ind = np.argmin(np.absolute(x))
    y_ind = np.argmin(np.absolute(y))
    z_ind = np.argmin(np.absolute(z))
    all_val = sdf_val.flatten()
    num_val = all_val[x_ind + y_ind * (sdf_res) + z_ind * (sdf_res) ** 2]
    return num_val > 0.0


def create_h5_sdf_pt(h5_file, sdf_file, flag_file, norm_obj_file,
                     centroid, m, sdf_res, num_sample, bandwidth, iso_val, max_verts, normalize, reduce=8):
    sdf_dict = get_sdf(sdf_file, sdf_res)
    ori_verts = np.asarray([0.0, 0.0, 0.0], dtype=np.float32).reshape((1, 3))
    # Nx3(x,y,z)

    print("ori_verts", ori_verts.shape)
    samplesdf, is_insideout = sample_sdf(num_sample, bandwidth, iso_val, sdf_dict, sdf_res, reduce)  # (N*8)x4 (x,y,z)
    if is_insideout:
        with open(flag_file, "w") as f:
            f.write("mid point sdf val > 0")
        print("insideout !!:", sdf_file)
    else:
        os.remove(flag_file) if os.path.exists(flag_file) else None
    print("samplesdf", samplesdf.shape)
    print("start to write", h5_file)

    norm_params = np.concatenate((centroid, np.asarray([m]).astype(np.float32)))
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('pc_sdf_original', data=ori_verts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('pc_sdf_sample', data=samplesdf.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.create_dataset('sdf_params', data=sdf_dict["param"], compression='gzip', compression_opts=4)
    f1.close()
    print("end writing", h5_file)
    command_str = "rm -rf " + norm_obj_file
    print("command:", command_str)
    os.system(command_str)
    command_str = "rm -rf " + sdf_file
    print("command:", command_str)
    os.system(command_str)



def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = trimesh.Trimesh(vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces)
    return mesh


def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384
    print("trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    print("[*] done!", model_file)

    mesh = as_mesh(mesh_list)  # from s2s
    if not isinstance(mesh, list):
        mesh_list = [mesh]

    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum += area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all = np.zeros((0, 3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        print("end sample surface")
        points_all = np.concatenate([points_all, points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")

    ori_mesh_list = trimesh.load_mesh(model_file, process=False)
    ori_mesh = as_mesh(ori_mesh_list)
    ori_mesh.vertices = (ori_mesh.vertices - centroid) / float(m)
    ori_mesh.export(obj_file)

    return obj_file, centroid, m


def create_one_sdf(sdfcommand, res, expand_rate, sdf_file, obj_file, indx, g=0.0):
    command_str = sdfcommand + " " + obj_file + " " + str(res) + " " + str(res) + \
                  " " + str(res) + " -s " + " -e " + str(expand_rate) + " -o " + str(indx) + ".dist -m 1"
    command_str += ' -c'
    if g > 0.0:
        command_str += " -g " + str(g)
    print("command:", command_str)
    os.system(command_str)
    command_str2 = "mv " + str(indx) + ".dist " + sdf_file
    print("command:", command_str2)
    os.system(command_str2)

if __name__ == "__main__":


    isosurface_dir = './isosurface/'
    sdf_cmd = f'{isosurface_dir}/computeDistanceField'
    mcube_cmd = f'{isosurface_dir}/computeMarchingCubes'
    lib_cmd = f'{isosurface_dir}/LIB_PATH'

    # set env variable
    os.environ[
        'LD_LIBRARY_PATH'] = f'$LD_LIBRARY_PATH:{isosurface_dir}:./isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7/'
    bandwidth = 0.1
    res = 256
    reduce = FLAGS.reduce
    expand_rate = 1.3
    iso_val = 0.003
    max_verts = 16384
    indx = 0
    g = 0.00
    norm = True
    model_file = FLAGS.model

    norm_mesh_sub_dir = './'
    sdf_file = './test.sdf'
    model_id = os.path.basename(model_file).split('.')[0]
    npy_file = os.path.join(FLAGS.save_path, f"{model_id}.npy")
    # npy_file = f'./sdf_dir_tmp/{model_id}.npy'
    start_time = time.time()
    norm_obj_file, centroid, m = get_normalize_mesh(model_file, norm_mesh_sub_dir)
    create_one_sdf(sdf_cmd, res, expand_rate, sdf_file, norm_obj_file, indx, g=g)

    sdf_dict = get_sdf(sdf_file, res)
    ori_verts = np.asarray([0.0, 0.0, 0.0], dtype=np.float32).reshape((1, 3))

    samplesdf, is_insideout = sample_sdf(sdf_dict, res, reduce)
    np.save(npy_file, samplesdf.squeeze(1).reshape(res//reduce, res//reduce, res//reduce))

    command_str = "rm -rf " + norm_obj_file
    print("command:", command_str)
    os.system(command_str)
    print(time.time()-start_time)

    command_str = "rm -rf " + sdf_file
    print("command:", command_str)
    os.system(command_str)

