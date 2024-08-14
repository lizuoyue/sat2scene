import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import tqdm, time
import json
import matplotlib; matplotlib.use("agg")
import matplotlib.pyplot as plt
import open3d as o3d
import torch
import sys
import cv2


def dep_to_cam_coord(dep):
    y = np.arange(dep.shape[0])
    x = np.arange(dep.shape[1])
    x, y = np.meshgrid(x, y)
    px = (x - 255.5) / 256
    py = (-y + 255.5) / 256
    pz = dep[..., 0]
    return np.dstack([px * pz, -py * pz, pz])

def cityscapes_classes():
    """Cityscapes class names for external use."""
    return [
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle'
    ]

def cityscapes_palette():
    """Cityscapes palette for external use."""
    return [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
            [0, 0, 230], [119, 11, 32]]

def xyz2lonlat(coord):
    # coord: N, 3
    dist = np.linalg.norm(coord, axis=-1)
    normed_coord = coord / dist[..., np.newaxis]
    lat = np.arcsin(normed_coord[:, 2]) # -pi/2 to pi/2
    lon = np.arctan2(normed_coord[:, 0], normed_coord[:, 1]) # -pi to pi
    return lon, lat

def xyz2uv(coord, img_h, img_w):
    # coord: N, 3
    lon, lat = xyz2lonlat(coord)
    lat /= (torch.pi / 2.0) # -1 to 1, map to h to 0
    lon /= torch.pi # -1 to 1, map to, 0 to w
    u = (-img_h * lat + img_h) / 2.0
    v = (img_w * lon + img_w) / 2.0
    return np.floor(np.stack([u, v], axis=-1)).astype(np.int32)

from vispy.util.transforms import rotate
def panorama_to_world(pano_yaw, tilt_yaw, tilt_pitch):
    """Convert d \in S^2 (direction of a ray on the panorama) to the world space."""
    axis = np.cross([np.cos(tilt_yaw), np.sin(tilt_yaw), 0], [0, 0, 1])
    R = (rotate(pano_yaw, [0, 0, 1]) @ rotate(tilt_pitch, axis))[:3, :3]
    return R


class HoliCityDataset(Dataset):
    def __init__(self, rootdir, split, since_month=None):
        self.rootdir = rootdir
        self.split = split

        filelist = np.genfromtxt(f'{rootdir}/split-all-v1-bugfix/{split}-middlesplit.txt', dtype=str)

        self.filelist = [f"{rootdir}/{f}" for f in filelist]
        self.filelist.sort()
        if since_month is not None:
            take_month = lambda s: s.split('/')[-2]
            self.filelist = [f for f in self.filelist if take_month(f) >= since_month]

        self.size = len(self.filelist)
        print(f'num {split}:', self.size)
        for i in range(self.size):
            assert(os.path.exists(f"{self.filelist[i][:-3]}.jpg"))
            assert(os.path.exists(f"{self.filelist[i]}_dpth.npz"))

        self.frames = []
        self.z_near = 0 # 0m
        self.z_far = 32 # 32m
        self.sem_path = "/cluster/project/cvg/zuoyue/ViT-Adapter/segmentation/holicity_4096x2048_seg"

    def __len__(self):
        return self.size
    
    def create_save_folder(self, save_folder):
        self.save_folder = save_folder
        os.system(f'mkdir -p {save_folder}')
    
    def check_locations(self):
        locations = []
        for idx in tqdm.tqdm(list(range(len(self)))):
            assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
            with open(f"{self.filelist[idx]}_camr.json") as cam_f:
                cam_info = json.load(cam_f)
                locations.append(np.array(cam_info["loc"]))
        locations = np.array(locations)
        plt.plot(locations[:, 0], locations[:, 1], 'o', alpha=0.1)
    
    def global_pts(self, depth, rot_mat):
        h, w = depth.shape
        func01 = lambda s: (np.arange(s) + 0.5) / s
        func11 = lambda s: func01(s) * 2 - 1
        lon, lat = np.meshgrid(func11(w) * np.pi, -func11(h) * np.pi / 2)

        # right x, inside y, up z
        direction = np.dstack(
            [
                np.cos(lat) * np.sin(lon),
                np.cos(lat) * np.cos(lon),
                np.sin(lat),
            ]
        )
        return (direction * depth[..., None]) @ rot_mat

    def save_data_pano_sem_sky(self, idx, downsample=1):

        assert(os.path.exists(f"{self.filelist[idx][:-3]}.jpg"))
        assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
        filename = os.path.basename(self.filelist[idx])[:-3]
        save_name = self.filelist[idx][:-3].split("/")[-1]

        if not os.path.exists(f"{self.sem_path}/{filename}.npz"):
            print(filename, "sem skip")
            return
        
        # assert(os.path.exists(f"{self.save_folder}/../sky_sem/{save_name}.png"))
        # try:
        #     Image.open(f"{self.save_folder}/../sky_sem/{save_name}.png").resize((512,256), Image.Resampling.LANCZOS)
        #     return
        # except:
        #     print(filename, "regen")
        #     return

        image = Image.open(f"{self.filelist[idx][:-3]}.jpg").resize(
            (4096, 2048), resample=Image.Resampling.LANCZOS
        )
        distance = np.load(f"{self.filelist[idx]}_dpth.npz")["depth"]

        sem = np.load(f"{self.sem_path}/{filename}.npz")["seg"]
        sky_mask = (sem == 10)#.astype(np.float32) * 255.0

        with open(f"{self.filelist[idx]}_camr.json") as cam_f:
            cam_info = json.load(cam_f)
        rot_mat = panorama_to_world(cam_info["pano_yaw"], cam_info["tilt_yaw"], cam_info["tilt_pitch"])

        h, w = 512, 1024
        func01 = lambda s: (np.arange(s) + 0.5) / s
        func11 = lambda s: func01(s) * 2 - 1
        lon, lat = np.meshgrid(func11(w) * np.pi, -func11(h) * np.pi / 2)

        # right x, inside y, up z
        direction = np.dstack(
            [
                np.cos(lat) * np.sin(lon),
                np.cos(lat) * np.cos(lon),
                np.sin(lat),
            ]
        )

        xy_dist = np.linalg.norm(self.global_pts(distance, rot_mat)[..., :2], axis=-1)
        sky_mask |= ((xy_dist >= 32.0) & ((sem < 5) | (sem == 8) | (sem == 9)))
        sky_mask = sky_mask.astype(np.float32) * 255.0
        
        pts = (direction @ np.linalg.inv(rot_mat)).reshape((-1, 3))
        uv = xyz2uv(pts, 2048, 4096)
        new_rgb = np.array(image)[uv[:, 0], uv[:, 1]]
        new_sem = sky_mask[uv[:, 0], uv[:, 1]][..., np.newaxis]
        new_image = np.concatenate([new_rgb, new_sem], axis=-1).reshape((h, w, 4)).astype(np.uint8)
        # Image.fromarray(new_image).save(f"{self.save_folder}/../sky_sem_v3/{save_name}.png")
        new_dist = xy_dist[uv[:, 0], uv[:, 1]].reshape((h, w)).astype(np.float32)
        to_vis = plt.get_cmap("viridis")(np.clip(new_dist / 64.0, 0.0, 1.0))[..., :3]
        Image.fromarray((to_vis * 255.0).astype(np.uint8)).save(f"{self.save_folder}/../sky_dep_v3/{save_name}.jpg")
        # np.savez_compressed(f"{self.save_folder}/../sky_dep_v3/{save_name}.npz", depth=new_dist)

    
    def save_data_pano_sem_resampling(self, idx, downsample=1):

        filename = os.path.basename(self.filelist[idx])[:-3]
        assert(os.path.exists(f"{self.filelist[idx][:-3]}.jpg"))
        assert(os.path.exists(f"{self.filelist[idx]}_dpth.npz"))
        assert(os.path.exists(f"{self.filelist[idx]}_camr.json"))
        if not os.path.exists(f"{self.sem_path}/{filename}.npz"):
            print(filename, "sem skip")
            return
        
        save_name = self.filelist[idx][:-3].split("/")[-1]

        if os.path.exists(f"{self.save_folder}/{save_name}.npz"):
            return
            # try:
            #     np.load(f"{self.save_folder}/{save_name}.npz")
            #     print(filename, "sem skip")
            #     return
            # except:
            #     print(filename, "regenerate")
            #     pass
        print(self.split, filename, "not exist and will generate")

        with open(f"{self.filelist[idx]}_camr.json") as cam_f:
            cam_info = json.load(cam_f)
        
        self.frames.append(np.array(cam_info["loc"]))

        depth = np.load(f"{self.filelist[idx]}_dpth.npz")["depth"][::downsample, ::downsample]

        image_org = Image.open(f"{self.filelist[idx][:-3]}.jpg")
        image = image_org.resize(
            depth.shape[::-1], resample=Image.Resampling.LANCZOS
        )
        image_org = np.array(image_org)
        sem = np.load(f"{self.sem_path}/{filename}.npz")["seg"]
        rot_mat = panorama_to_world(cam_info["pano_yaw"], cam_info["tilt_yaw"], cam_info["tilt_pitch"])

        h, w = depth.shape
        func01 = lambda s: (np.arange(s) + 0.5) / s
        func11 = lambda s: func01(s) * 2 - 1
        lon, lat = np.meshgrid(func11(w) * np.pi, -func11(h) * np.pi / 2)

        # right x, inside y, up z
        direction = np.dstack(
            [
                np.cos(lat) * np.sin(lon),
                np.cos(lat) * np.cos(lon),
                np.sin(lat),
            ]
        )

        pts = (direction * depth[..., None]).reshape((-1, 3)) @ rot_mat
        xy_dist = np.linalg.norm(pts[:, :2], axis=-1)

        # Update depth
        depth = pts[:, -1].reshape((h, w))
        gray_dep = np.clip(depth * 8.0, 0.0, 255.0).astype(np.uint8)  # Only 0-32m

        # Image.fromarray(gray_dep).save(f"{self.save_folder}/{save_name}_dep.jpg")
        min_th, max_th = 32, 64
        edges = cv2.Canny(gray_dep, min_th, max_th) > 0
        non_edges = ~edges.reshape((-1))

        # Image.fromarray((edges * 255).astype(np.uint8)).save(f"{self.save_folder}/{save_name}_depth_edge.png")

        from scipy.ndimage.filters import maximum_filter, minimum_filter
        from scipy.ndimage.morphology import generate_binary_structure
        depth_local_max = maximum_filter(depth, footprint=generate_binary_structure(2, 2)) == depth
        depth_local_min = minimum_filter(depth, footprint=generate_binary_structure(2, 2)) == depth

        mask = ((self.z_near <= xy_dist) & (xy_dist <= self.z_far)).reshape((-1))
        mask &= ~depth_local_max.reshape((-1))
        mask &= ~depth_local_min.reshape((-1))

        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(pts.reshape((2048, 4096, 3))[::4, ::4].reshape((-1, 3)))
        # pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30))
        # pc.normalize_normals()
        # normal = np.asarray(pc.normals).reshape((512, 1024, 3))
        # print(normal.min(), normal.max(), normal.dtype)
        # cm_rainbow = matplotlib.cm.get_cmap('rainbow')
        # pseudo_dep = cm_rainbow(depth / 32.0)
        # print(pseudo_dep.shape, pseudo_dep.min(), pseudo_dep.max(), pseudo_dep.dtype)
        # Image.fromarray(((normal * 0.5 + 0.5) * 255.0).astype(np.uint8)).convert('RGB').save(f"{self.save_folder}/{save_name}_nml.jpg")
        # quit()

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        idx = w * y + x
        idx_up_left = idx[:-1]
        idx_up_right = np.roll(idx_up_left, -1, axis=1)
        idx_down_left = idx[1:]
        idx_down_right = np.roll(idx_down_left, -1, axis=1)

        if True:
            upper = np.stack([idx_up_left, idx_up_right, idx_down_left], axis=-1).reshape((-1, 3))
            lower = np.stack([idx_down_left, idx_up_right, idx_down_right], axis=-1).reshape((-1, 3))
            all_tri = np.concatenate([upper, lower], axis=0)
            # ---
            # |/|
            # ---
        else:
            upper = np.stack([idx_up_left, idx_up_right, idx_down_right], axis=-1).reshape((-1, 3))
            lower = np.stack([idx_down_left, idx_up_left, idx_down_right], axis=-1).reshape((-1, 3))
            all_tri = np.concatenate([upper, lower], axis=0)
            # ---
            # |\|
            # ---

        tri_ok = mask[all_tri].all(axis=-1)
        scene_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(pts),
            triangles=o3d.utility.Vector3iVector(all_tri[tri_ok]),
        )
        scene_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(image).reshape((-1, 3)) / 255.0)
        scene_mesh.compute_triangle_normals()

        origin_tri_dist = np.abs(np.einsum(
            "nc, npc -> np",
            np.asarray(scene_mesh.triangle_normals),
            pts[np.asarray(scene_mesh.triangles)],
        ).mean(axis=1))  # should get similar result each row
        origin_tri_cen_dist = np.linalg.norm(pts[np.asarray(scene_mesh.triangles)].mean(axis=1), axis=-1)
        origin_tri_cen_dist_valid = origin_tri_cen_dist > 1e-6
        origin_tri_plane_sin = origin_tri_dist / np.clip(origin_tri_cen_dist, a_min=1e-6, a_max=None)
        # plt.hist(origin_tri_dist, bins=500, range=(0, 1))
        # plt.ylim(0, 10000)
        # plt.savefig("aaa.png")
        # quit()
        # hist, bin_edges = np.histogram(origin_tri_dist, bins=640, range=(0, 32))
        # for l, r, n in zip(bin_edges[:-1], bin_edges[1:], hist):
        #     print(f"[{l}, {r}), {n}")
        #     input()
        # perm = np.arange(origin_tri_dist.shape[0])
        # np.random.shuffle(perm)
        # perm = perm[:(origin_tri_dist.shape[0] // 1)]
        # tri_pts_mean = pts[np.asarray(scene_mesh.triangles)].mean(axis=1)[perm]
        # origin_tri_dist_color = plt.get_cmap("viridis")(np.clip(origin_tri_dist, 0.0, 1.0))[perm]
        # with open(f"{self.save_folder}/{save_name}_tri_dist.txt", "w") as f:
        #     for (x, y, z), (r, g, b, _) in zip(tri_pts_mean, (origin_tri_dist_color * 255.0).astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        # quit()

        def angle(triangles, idx):
            # The cross product of two sides is a normal vector
            v1 = triangles[:, (idx + 1) % 3] - triangles[:, idx]
            v2 = triangles[:, (idx + 2) % 3] - triangles[:, idx]
            norm1 = np.linalg.norm(v1, axis=-1)
            norm2 = np.linalg.norm(v2, axis=-1)
            norm12 = norm1 * norm2
            result = np.arccos(
                np.clip(
                    np.sum(v1 * v2, axis=-1) / np.clip(norm12, 1e-6, None),
                    0.0, 1.0,
                )
            ) # in range [0, pi]
            result[norm12 <= 1e-6] *= 0
            return result
        
        def angles(triangles):
            return np.stack([angle(triangles, 0), angle(triangles, 1), angle(triangles, 2)], axis=1)

        # TODO: add more constraint?
        not_through_origin = origin_tri_cen_dist_valid & (origin_tri_plane_sin > 0.03)
        not_edge = non_edges[np.asarray(scene_mesh.triangles)].all(axis=-1)
        not_sharp = (angles(pts[np.asarray(scene_mesh.triangles)]) >= (3.0 / 180.0 * np.pi)).all(axis=-1)

        scene_mesh.triangles = o3d.utility.Vector3iVector(all_tri[tri_ok][
            not_edge & (not_sharp | not_through_origin)
        ])

        # o3d.io.write_triangle_mesh(f"{filename}.ply", scene_mesh)
        # return

        def normal(triangles):
            # The cross product of two sides is a normal vector
            return np.cross(triangles[:,1] - triangles[:,0], 
                            triangles[:,2] - triangles[:,0], axis=1)

        def surface_area(triangles):
            # The norm of the cross product of two sides is twice the area
            return np.linalg.norm(normal(triangles), axis=1) / 2

        area = np.sum(surface_area(pts[np.asarray(scene_mesh.triangles)]))
        n_pts = int(area * 400)
        if n_pts < (200000):
            print(save_name, "skip")
            return
        else:
            print(save_name, n_pts)

        pc = scene_mesh.sample_points_poisson_disk(n_pts)
        pc.estimate_normals()
        pc.normalize_normals()
        pc_ground_mask = np.abs(np.asarray(pc.normals)[:, 2]) > 0.98
        pc_non_ground_mask = ~pc_ground_mask

        src_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(pts)
        dist = pc.compute_point_cloud_distance(src_pc)

        uv = xyz2uv(np.asarray(pc.points) @ np.linalg.inv(rot_mat), sem.shape[0], sem.shape[1])
        pc_semantics = sem[
            np.clip(uv[:, 0], 0, sem.shape[0] - 1),
            np.clip(uv[:, 1], 0, sem.shape[1] - 1),
        ]

        np.savez_compressed(
            f"{self.save_folder}/{save_name}.npz",
            coord=np.asarray(pc.points),
            color=(np.asarray(pc.colors) * 255.0).astype(np.uint8),
            dist=dist,
            sem=pc_semantics,
            geo_is_not_ground=pc_non_ground_mask,
        )
        return

        # with open(f"{self.save_folder}/{save_name}.txt", "w") as f:
        #     for (x, y, z), (r, g, b) in zip(np.asarray(pc.points), (np.asarray(pc.colors) * 255.0).astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        # pc_semantics_colored = np.array(cityscapes_palette())[pc_semantics]
        # with open(f"{self.save_folder}/{save_name}_sem.txt", "w") as f:
        #     for (x, y, z), (r, g, b) in zip(np.asarray(pc.points), pc_semantics_colored.astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        # pc_cls_colored = np.array(cityscapes_palette())[pc_non_ground_mask.astype(np.int32) + 1]
        # with open(f"{self.save_folder}/{save_name}_binary.txt", "w") as f:
        #     for (x, y, z), (r, g, b) in zip(np.asarray(pc.points), pc_cls_colored.astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))

        # pc_dist_colored = plt.get_cmap("viridis")(np.clip(dist, 0.0, 0.5) * 2)
        # with open(f"{self.save_folder}/{save_name}_dist.txt", "w") as f:
        #     for (x, y, z), (r, g, b, _) in zip(np.asarray(pc.points), (pc_dist_colored * 255.0).astype(np.uint8)):
        #         f.write('%.3lf %.3lf %.3lf %d %d %d\n' % (x, y, z, r, g, b))
        
        # return



if __name__ == "__main__":

    args_li = []
    dataset = {}
    for phase in ["train", "valid", "test"]:
        dataset[phase] = HoliCityDataset("/cluster/project/cvg/zuoyue/HoliCity", phase, since_month="2008-07")
        dataset[phase].create_save_folder(f"/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_density400/{phase}")
        args_li.extend([(phase, idx) for idx in range(len(dataset[phase]))])

    import multiprocessing
    def func(phase, idx):
        # dataset[phase].save_data_pano_sem_sky(idx, downsample=1)
        dataset[phase].save_data_pano_sem_resampling(idx, downsample=1)

    cpu_count = multiprocessing.cpu_count()
    if len(os.sched_getaffinity(0)) < cpu_count:
        try:
            os.sched_setaffinity(0, range(cpu_count))
        except OSError:
            print('Could not set affinity')

    n = len(os.sched_getaffinity(0))
    print('Using', n, 'processes for the pool')
    with multiprocessing.Pool(n) as pool:
        with tqdm.tqdm(total=len(args_li)) as pbar:
            async_results = []
            for args in args_li:
                async_results.append(
                    pool.apply_async(func, args=args, callback=lambda _: pbar.update())
                )
            results = [async_result.get() for async_result in async_results]


    # Below for single CPU run
    # for phase in ["train", "valid", "test"]:
    #     dataset = HoliCityDataset("/cluster/project/cvg/zuoyue/HoliCity", phase, since_month="2008-07")
    #     dataset.create_save_folder(f"/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400_all/{phase}")
    #     dataset.save_data_pano_sem_sky(0, downsample=1)
    # #     dataset.check_locations()
    # # plt.savefig("split_locations.png")
    # # train valid test 2018-01

    # dataset = HoliCityDataset("/cluster/project/cvg/zuoyue/HoliCity", sys.argv[1], since_month="2008-07")
    # dataset.create_save_folder(f"/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400_all/{sys.argv[1]}")
    # dataset.create_save_folder(f"/cluster/project/cvg/zuoyue/holicity_point_cloud/4096x2048_resample_400_all/sky_sem")
    # for i in tqdm.tqdm(list(range(int(sys.argv[2]), int(sys.argv[3])))):  # len(dataset)
    #     dataset.save_data_pano_sem_resampling(i, downsample=1)
    #     dataset.save_data_pano_sem_sky(i, downsample=1)
