"""Project a saved SMPL mesh onto its matching camera frame (weak perspective)."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from camera_auto_fast import faces_from_smpl

p=argparse.ArgumentParser()
p.add_argument('--snapshot',required=True,type=Path)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
p.add_argument('--render-only',action='store_true',help='save overlay without opening a window')
args=p.parse_args()
cv2.setNumThreads(1)
data=np.load(args.snapshot/'prediction.npz')
meta=json.loads((args.snapshot/'metadata.json').read_text())
frame=cv2.imread(str(args.snapshot/'frame.jpg'))
v=data['vertices'];s,tx,ty=data['camera']
left,top,side,_=meta['crop_box_xywh']
xy=s*(v[:,:2]+np.array([tx,ty]))
pixels=np.round((xy+1)*side/2+[left,top]).astype(np.int32)
faces=faces_from_smpl(args.root/'source/data/smpl_neutral_f32.bin').astype(np.int32)
tri=v[faces]
normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
colors=((.4+.6*np.abs(normals@np.array([0.,0.,1.])))[:,None]*[230,180,100]).astype(np.uint8)
surface=frame.copy(); mask=np.zeros(frame.shape[:2],np.uint8)
# The viewer looks along positive model Z; nearer triangles have smaller Z.
for i in np.argsort(tri[:,:,2].mean(1))[::-1]:
    cv2.fillConvexPoly(surface,pixels[faces[i]],tuple(int(c) for c in colors[i]))
    cv2.fillConvexPoly(mask,pixels[faces[i]],255)
overlay=frame.copy()
blended=cv2.addWeighted(frame,.35,surface,.65,0)
overlay[mask>0]=blended[mask>0]
cv2.imwrite(str(args.snapshot/'mesh_overlay.jpg'),overlay)
panel=np.concatenate([frame,overlay],axis=1)
cv2.putText(panel,'Captured frame                       SMPL projected overlay',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,240,240),2)
cv2.imwrite(str(args.snapshot/'mesh_overlay_comparison.jpg'),panel)
if args.render_only:
    print('OVERLAY_SAVED',args.snapshot,flush=True)
    raise SystemExit(0)
title='ODROID fresh capture - mesh overlay | Q: close'
cv2.namedWindow(title,cv2.WINDOW_NORMAL)
cv2.resizeWindow(title,1280,480)
cv2.imshow(title,panel)
print('OVERLAY_DISPLAYED',args.snapshot, 'camera=',data['camera'].tolist(),flush=True)
while cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)>=1:
    if cv2.waitKey(100)&255 in (ord('q'),27):break
cv2.destroyAllWindows()
