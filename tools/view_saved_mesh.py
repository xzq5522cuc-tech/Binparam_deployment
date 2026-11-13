"""HDMI viewer for board-generated SMPL predictions; rerender only on interaction."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from camera import faces_from_smpl

p=argparse.ArgumentParser()
p.add_argument('--snapshot',required=True,type=Path)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
args=p.parse_args()
cv2.setNumThreads(1)
data=np.load(args.snapshot/'prediction.npz')
vertices=data['vertices'].copy()
assert vertices.shape==(6890,3) and np.isfinite(vertices).all()
vertices-=vertices.mean(0)
faces=faces_from_smpl(args.root/'source/data/smpl_neutral_f32.bin').astype(np.int32)
frame=cv2.imread(str(args.snapshot/'frame.jpg'))
meta=json.loads((args.snapshot/'metadata.json').read_text())
action=meta.get('action') or {}
pose_names={'LEFT_ARM_UP':'LEFT ARM UP', 'RIGHT_ARM_UP':'RIGHT ARM UP',
            'STANDING':'STANDING', 'SQUAT':'SQUAT', 'BOTH_ARMS_UP':'BOTH ARMS UP'}
pose_label=action.get('label','UNKNOWN')
pose_color=(100,240,130) if pose_label in pose_names else (0,220,255)
pose_text='Saved pose: '+pose_names.get(pose_label,'UNKNOWN')
if pose_label=='CONFIRMING':
    candidate=pose_names.get(action.get('candidate'),'UNKNOWN')
    required=action.get('required_predictions',3)
    progress=min(action.get('consecutive',0),required)
    pose_text=f'Confirming: {candidate} ({progress}/{required})'
elif not action:
    pose_text='Saved pose: UNAVAILABLE (older snapshot)'
title='ODROID local 3D mesh - saved camera reconstruction'
yaw,pitch,zoom,wire=0.,0.,1.,False
pan=np.zeros(2,dtype=float)
mouse={'drag':None,'last':(0,0),'quit':False}
dirty=True
buttons=[(650,620,785,660,'Return'),(795,620,930,660,'Reset'),(940,620,1105,660,'Wire / solid')]

def reset_view():
    global yaw,pitch,zoom,dirty
    yaw,pitch,zoom=0.,0.,1.
    pan[:]=0
    dirty=True

def on_mouse(event,x,y,flags,userdata):
    global yaw,pitch,zoom,wire,dirty
    if event==cv2.EVENT_LBUTTONUP:
        mouse['drag']=None
        for index,(x1,y1,x2,y2,label) in enumerate(buttons):
            if x1<=x<=x2 and y1<=y<=y2:
                if index==0:mouse['quit']=True
                elif index==1:reset_view()
                else:wire=not wire;dirty=True
                return
    elif event==cv2.EVENT_RBUTTONUP:
        mouse['drag']=None
    elif event in (cv2.EVENT_LBUTTONDOWN,cv2.EVENT_RBUTTONDOWN):
        if 640<=x<1120 and 0<=y<550:
            mouse['drag']='rotate' if event==cv2.EVENT_LBUTTONDOWN else 'pan'
            mouse['last']=(x,y)
    elif event==cv2.EVENT_MOUSEMOVE and mouse['drag'] is not None:
        required=cv2.EVENT_FLAG_LBUTTON if mouse['drag']=='rotate' else cv2.EVENT_FLAG_RBUTTON
        if not flags & required:
            mouse['drag']=None
            return
        dx,dy=x-mouse['last'][0],y-mouse['last'][1]
        mouse['last']=(x,y)
        if mouse['drag']=='rotate':
            yaw+=dx*.5
            pitch=float(np.clip(pitch-dy*.5,-89,89))
        else:pan[:]+=[dx,dy]
        dirty=True
    elif event==cv2.EVENT_MOUSEWHEEL and 640<=x<1120 and 0<=y<550:
        delta=(flags>>16)&0xffff
        if delta>=32768:delta-=65536
        if delta:
            zoom=float(np.clip(zoom*(1.12 if delta>0 else 1/1.12),.25,4.))
            dirty=True

def render():
    a,b=np.deg2rad([yaw,pitch])
    ry=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    rx=np.array([[1,0,0],[0,np.cos(b),-np.sin(b)],[0,np.sin(b),np.cos(b)]])
    v=vertices@(rx@ry).T
    xy=v[:,:2].copy()
    xy-=(xy.min(0)+xy.max(0))/2
    scale=min(400/max(np.ptp(xy[:,0]),.1),450/max(np.ptp(xy[:,1]),.1))*zoom
    pixels=np.round(xy*scale+[240,275]+pan).astype(np.int32)
    panel=np.full((550,480,3),245,np.uint8)
    if wire:
        cv2.polylines(panel,pixels[faces],True,(145,95,35),1,cv2.LINE_8)
    else:
        tri=v[faces]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
        light=np.array([.3,-.4,1.]);light/=np.linalg.norm(light)
        colors=((.35+.65*np.abs(normals@light))[:,None]*[220,175,110]).astype(np.uint8)
        for i in np.argsort(tri[:,:,2].mean(1))[::-1]:
            cv2.fillConvexPoly(panel,pixels[faces[i]],tuple(int(c) for c in colors[i]))
    canvas=np.full((670,1120,3),28,np.uint8)
    h,w=frame.shape[:2]; factor=min(620/w,500/h)
    preview=cv2.resize(frame,(round(w*factor),round(h*factor)))
    canvas[50:50+preview.shape[0],:preview.shape[1]]=preview
    canvas[:550,640:1120]=panel
    cv2.putText(canvas,pose_text,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.60,pose_color,2)
    lines=[f"ON-BOARD | 6890 vertices | {len(faces)} triangles | inference {meta.get('inference_ms',0):.1f} ms",
           'Mouse on mesh: left drag rotate | right drag move | wheel zoom']
    for i,line in enumerate(lines):
        cv2.putText(canvas,line,(12,575+i*25),cv2.FONT_HERSHEY_SIMPLEX,.52,(235,235,235),1)
    # Keep the bottom row available for mouse-only navigation.
    cv2.rectangle(canvas,(0,615),(1120,670),(28,28,28),-1)
    cv2.putText(canvas,'A/D W/S +/- F R Q also work',(12,645),cv2.FONT_HERSHEY_SIMPLEX,.5,(210,210,210),1)
    for x1,y1,x2,y2,label in buttons:
        cv2.rectangle(canvas,(x1,y1),(x2,y2),(45,105,65),-1)
        cv2.putText(canvas,label,(x1+12,y1+27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return canvas

cv2.namedWindow(title,cv2.WINDOW_NORMAL)
cv2.resizeWindow(title,1120,670)
cv2.setMouseCallback(title,on_mouse)
preview_saved=False
try:
    while True:
        if dirty:
            start=time.monotonic();canvas=render()
            if not preview_saved:
                cv2.imwrite(str(args.snapshot/'hdmi_viewer_preview.jpg'),canvas)
                preview_saved=True
            cv2.imshow(title,canvas)
            print(f'DISPLAY_UPDATED yaw={yaw} pitch={pitch} wire={wire} render_ms={(time.monotonic()-start)*1000:.1f}',flush=True)
            dirty=False
        key=cv2.waitKey(50)&255
        if mouse['quit'] or key in (ord('q'),27) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
            break
        if key==ord('a'):yaw-=15
        elif key==ord('d'):yaw+=15
        elif key==ord('w'):pitch-=15
        elif key==ord('s'):pitch+=15
        elif key in (ord('+'),ord('=')):zoom=min(2.,zoom*1.1)
        elif key==ord('-'):zoom=max(.4,zoom/1.1)
        elif key==ord('f'):wire=not wire
        elif key==ord('r'):reset_view()
        else:continue
        dirty=True
finally:
    cv2.destroyAllWindows()
