import numpy as np 
import pdb 
import json 
import cv2 
import os 
import config
import torch 
from models import hmr, SMPL, smpl
import os.path as osp 
import sys 
import torch 
import os
from torchvision.transforms import Normalize
import constants
from utils.imutils import crop, transform, flip_img, flip_pose, flip_kp, transform, rot_aa
import pdb 
from utils.geometry import batch_rodrigues
from utils.renderer import Renderer

# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def make_skeleton(kps, img):
    color = (51, 0, 0)
    spine_color = (0, 102, 204)


    color_list = ['maroon','red','salmon','darkslateblue','blue','navy','sienna','orangered','darkorange',
            'royalblue','dodgerblue','steelblue','chartreuse','lightgreen','black','gray','lightgray',
            'forestgreen','limegreen','lime','seagreen','green','darkgreen','palegreen']

    thickness = 2
    linetype = cv2.LINE_AA
    
    
    # points 
    # Right Ankle, Left Ankle 
    cv2.line(img, tuple(kps[0]), tuple(kps[0]), (25, 0, 51), thickness * 3, linetype)
    cv2.line(img, tuple(kps[5]), tuple(kps[5]), (25, 0, 51), thickness * 3, linetype)


    # Right Wrist, Left Wrist        # right -> Yellow 
    cv2.line(img, tuple(kps[6]), tuple(kps[6]), (51, 51, 0), thickness * 3, linetype)
    cv2.line(img, tuple(kps[11]), tuple(kps[11]), (102, 0, 204), thickness * 3, linetype)

    # lines 
    # from ankle to hip 
    cv2.line(img, tuple(kps[0]), tuple(kps[1]), color, thickness, linetype)            # cv2.line(img, RAnkle, RKnee, color, thickness, linetype)
    cv2.line(img, tuple(kps[1]), tuple(kps[2]), color, thickness, linetype)            # cv2.line(img, RKnee, RHip, color, thickness, linetype)
    cv2.line(img, tuple(kps[5]), tuple(kps[4]), color, thickness, linetype)            # cv2.line(img, LAnkle, LKnee, color, thickness, linetype)
    cv2.line(img, tuple(kps[4]), tuple(kps[3]), color, thickness, linetype)            # cv2.line(img, LKnee, LHip, color, thickness, linetype)

    # between hip 
    cv2.line(img, tuple(kps[2]), tuple(kps[3]), color, thickness, linetype)            # cv2.line(img, RHip, LHip, color, thickness, linetype)

    # from wrist to sholder 
    cv2.line(img, tuple(kps[6]), tuple(kps[7]), color, thickness, linetype)            # cv2.line(img, RWrist, RElbow, color, thickness, linetype)
    cv2.line(img, tuple(kps[7]), tuple(kps[8]), color, thickness, linetype)            # cv2.line(img, RElbow, RShoulder, color, thickness, linetype)
    cv2.line(img, tuple(kps[9]), tuple(kps[10]), color, thickness, linetype)            # cv2.line(img, LWrist, LElbow, color, thickness, linetype)
    cv2.line(img, tuple(kps[10]), tuple(kps[11]), color, thickness, linetype)            # cv2.line(img, LElbow, LShoulder, color, thickness, linetype)

    # between shoulder 
    cv2.line(img, tuple(kps[8]), tuple(kps[9]), color, thickness, linetype)            # cv2.line(img, RShoulder, LShoulder, color, thickness, linetype)

    # R-R, L-L
    cv2.line(img, tuple(kps[2]), tuple(kps[8]), color, thickness, linetype)            # cv2.line(img, RHip, RShoulder, color, thickness, linetype)
    cv2.line(img, tuple(kps[3]), tuple(kps[9]), color, thickness, linetype)            # cv2.line(img, LHip, LShoulder,, color, thickness, linetype)
    return img


def rgb_processing(rgb_img, center, scale, rot=0):
    """Process rgb image and do augmentation."""
    rgb_img = crop(rgb_img, center, scale, 
                    [constants.IMG_RES, constants.IMG_RES], rot=rot)
    # (3,224,224),float,[0,1]
    rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    return rgb_img


def j2d_processing(kp, center, scale, r=0):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    nparts = kp.shape[0]
    orig_kp = kp.copy()
    for i in range(nparts):
        kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                [constants.IMG_RES, constants.IMG_RES], rot=r)
    # convert to normalized coordinates
    kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
    kp = kp.astype('float32')
    return orig_kp, kp


# normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)

smpl = smpl.SMPL(config.SMPL_MODEL_DIR,
                    batch_size=1,
                    create_transl=False).cuda()

action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

# filename 어떻게 저장할지도 생각할 것 -> 기존 포맷에 맞게! 

# 1. annot-image
# 2. annot-annotations
# 3. images-sample image from file_name
# 4. smpl annot - annot via fram_idx 

import os.path as osp 
import cv2 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

summary_writer = SummaryWriter('tensorboard_test')
torch.set_num_threads(2)

images_path = osp.join('/home', 'ubuntu', 'data', 'Human36M', 'images')
annot_path = osp.join('/home', 'ubuntu', 'data', 'Human36M', 'annotations')
subject_list = []
scaleFactor = 1.2 
renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=224., faces=smpl.faces)
subject_list = [1, 5, 6, 7, 8, 9, 11]
invalid = 0
valid = 0
total = 0


imgname_list = []
center_list = []
scale_list = []
part_list = []
pose_list = []
shape_list = []
S_list = []


for sbj in subject_list:        # 일단 한 subj 안에 들어가면
    print(f'sbj {sbj} processing!')
    sbj_anno_path = osp.join(annot_path, f'Human36M_subject{sbj}_data.json')
    sbj_smpl_path = osp.join(annot_path, f'Human36M_subject{sbj}_smpl_param.json')
    sbj_camera_path = osp.join(annot_path, f'Human36M_subject{sbj}_camera.json')
    sbj_joints_path = osp.join(annot_path, f'Human36M_subject{sbj}_joint_3d.json')

    annot_list = json.load(open(sbj_anno_path, 'r'))
    smpl_list = json.load(open(sbj_smpl_path, 'r'))
    camera_list = json.load(open(sbj_camera_path, 'r'))
    joints_list = json.load(open(sbj_joints_path, 'r'))

    for idx in range(len(annot_list['images'])):      # 그 안에 있는 data annotation 전부 확보 가능 
        sample_file_name = annot_list['images'][idx]['file_name']     # folder / filename <- same as "image directory" ~/data/Human36M/images/folder/filename
        subject = annot_list['images'][idx]['subject']
        action_idx = annot_list['images'][idx]['action_idx']
        subaction_idx = annot_list['images'][idx]['subaction_idx']
        frame_idx = annot_list['images'][idx]['frame_idx']
        cam_idx = annot_list['images'][idx]['cam_idx']
        R = np.array(camera_list[f'{cam_idx}']['R'])
        t = np.array(camera_list[f'{cam_idx}']['t'])
        f = np.array(camera_list[f'{cam_idx}']['f'])
        c = np.array(camera_list[f'{cam_idx}']['c'])

        bbox = annot_list['annotations'][idx]['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200

        image_idx = annot_list['annotations'][idx]['image_id']

        joint_world = np.array(joints_list[str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)

        joint_cam = world2cam(joint_world, R, t)        # <-h36m은 이거로 비교해야하나..? #<- 우선 S로 저장해보기  #<- 이미지이름은 
        joint_img = cam2pixel(joint_cam, f, c)

        joint_img, h36m_kps = j2d_processing(joint_img, center, scale)

        # 1 ~ 6 check again with another sample
        # 0 - Pelvis, 1 - RHip, 2 - RKnee, 3 - RAnkle, 4 - LHip, 5 - LKnee, 6 - KAnkle, 7 - Spine, 8 - Neck, 9 - Jaw, 
        # 10 - Head, 11 - LShoulder, 12 - LElbow, 13-LWrist, 14 - RShoulder, 15 - RElbow, 16 - RWrist
        kps = np.zeros((24, 3))
        kps_final = np.zeros((24, 3))

        global_idx = [14, 2, 1, 0, 3, 4, 5, 16, 12, 17, 18, 9, 10, 11, 8, 7 ,6]
        not_human_idx = list(set())

        kps[global_idx] = h36m_kps
        kps_final[global_idx] = joint_img

        imgname = osp.join(images_path, sample_file_name)      # image path okey 
        save_imgname = '/'.join(imgname.split('/')[5:])

        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        image = rgb_processing(img, center, scale)
        input_img = torch.from_numpy(image).float().unsqueeze(0)       # image를 모델에 넣어줄게 아니라면 굳이 Normalize하지는 않는다. 
        # input_img = normalize_img(img)            # render할 때도 바로 이미지 위에 얹어서 넣어줌 

        try:
            smpl_param = smpl_list[f'{action_idx}'][f'{subaction_idx}'][f'{frame_idx}']
            pose = torch.from_numpy(np.array(smpl_param['pose']).reshape(24, 3)).cuda().float()
            rotmat = batch_rodrigues(pose.view(-1, 3)).view([1, -1, 3, 3])
            rotmat[:, 0] = torch.matmul(torch.tensor(R).cuda().float(), rotmat[:, 0]).clone()
            betas = torch.from_numpy(np.array(smpl_param['shape'])).cuda().float().reshape(1, 10)
            pred_output = smpl(betas=betas, body_pose=rotmat[:, 1:], global_orient=rotmat[:, 0].unsqueeze(1), pose2rot=False)
            pred_joints = pred_output.joints[:, 25:, :].squeeze()
            pred_vertices = pred_output.vertices            
            # make skeleton 
            images_cpu = input_img.cpu().squeeze()
            # for one image sample 
            images = images_cpu.numpy().transpose((1, 2, 0))
            images = np.ascontiguousarray(images)

            kps = ((kps + 1) * 112).astype(np.int32)

            # confidence = np.where(kps[:, :2] >= 224 or kps[:, :2] < 0)
            confidence1 = list(set(np.where(kps[:, :2] >= 224)[0]))
            confidence2 = list(set(np.where(kps[:, :2] < 0)[0]))
            confidence = confidence1 + confidence2
            kps_final[confidence, 2] = 0
            kps_final[list(set(list(range(len(kps)))) - set(confidence)), 2] = 1
            kps_final[list(set(list(range(24))) - set(global_idx)), 2] = 0

            kps = ((kps + 1) * 112).astype(np.int32)

            empty_kps = np.zeros((25, 3))
            kps_final = np.concatenate((empty_kps, kps_final), axis = 0) # <- add another 25 dimension at top 
            # S = np.concatenate((np.array(pred_joints.cpu()), kps_final[25:, -1].reshape(len(pred_joints), 1)), axis = -1)

            images_pred = renderer.visualize_tb(pred_vertices, pred_cam_t, input_img)        

            imgname_list.append(imgname)
            center_list.append(np.array(center))
            scale_list.append(np.float(scale))
            part_list.append(kps_final)
            pose_list.append(np.array(rotmat.squeeze().cpu()))
            shape_list.append(np.array(betas.squeeze().cpu()))
            S_list.append(joint_cam)

            imgs = []
            skeleton = torch.from_numpy(np.transpose(make_skeleton(kps[:, :2], images), (2, 0, 1))).float()
            imgs.append(skeleton)
            imgs = make_grid(imgs, nrow=1)

            summary_writer.add_image('pred_shape', imgs, valid)

            valid += 1
            print(f'valid {valid}')

        except:
            pdb.set_trace()
            invalid += 1
            print(f'invalid_img {imgname}')
        total += 1
        
print(f'total {total}')
cam_list = np.ones((valid, 3))
has_smpl_list = np.ones(valid, )
print('saved!')
np.savez('h36m3D_I2L.npz', imgname = imgname_list, \
                                        center = center_list, \
                                        scale = scale_list, \
                                        part = part_list, \
                                        pose = pose_list, \
                                        shape = shape_list, \
                                        cam = cam_list, \
                                        has_smpl = has_smpl_list, \
                                        S = S_list)


